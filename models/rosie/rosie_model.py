from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from spora_bench.utils.virtual_staining_utils import (
    IMAGENET_MEAN, IMAGENET_STD, build_marker_index, denorm_he_uint8,
)
from spora_bench.wrapper import SporaModelWrapper

# ROSIE background-skip threshold (mean patch value above which a patch is "white")
WHITE_THRESHOLD = 220

# ROSIE: 50 channels, fixed order (HF model card).
ROSIE_CHANNELS = [
    "DAPI", "CD45", "CD68", "CD14", "PD1", "FoxP3", "CD8", "HLA-DR", "PanCK", "CD3e",
    "CD4", "aSMA", "CD31", "Vimentin", "CD45RO", "Ki67", "CD20", "CD11c", "Podoplanin", "PDL1",
    "GranzymeB", "CD38", "CD141", "CD21", "CD163", "BCL2", "LAG3", "EpCAM", "CD44", "ICOS",
    "GATA3", "Gal3", "CD39", "CD34", "TIGIT", "ECad", "CD40", "VISTA", "HLA-A", "MPO",
    "PCNA", "ATM", "TP63", "IFNg", "Keratin8/18", "IDO1", "CD79a", "HLA-E", "CollagenIV", "CD66",
]


class _ROSIEPatchDataset(Dataset):
    """Flattens (tile, grid-point) into one index space and yields 128px patches
    preprocessed exactly as ROSIE's evaluate.py (ToTensor -> Resize224 -> imagenet norm)."""

    def __init__(self, he_images: List[np.ndarray], patch: int, grid_stride: int, exclude_bg: bool):
        import torchvision.transforms as T
        self.imgs = he_images  # list of HWC uint8
        self.ps = patch // 2
        self.patch = patch
        self.tf_all = T.Compose([T.ToTensor(), T.Resize(224, antialias=True)])
        self.tf_norm = T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
        ch = max(1, grid_stride // 2)
        self.index = []  # (image_idx, x, y)
        for ii, img in enumerate(self.imgs):
            H, W = img.shape[:2]
            for y in range(0, H, grid_stride):
                for x in range(0, W, grid_stride):
                    if exclude_bg:
                        ys, ye = max(0, y - ch), min(H, y + ch)
                        xs, xe = max(0, x - ch), min(W, x + ch)
                        if float(img[ys:ye, xs:xe].mean()) >= WHITE_THRESHOLD:
                            continue
                    self.index.append((ii, x, y))

    def __len__(self):
        return len(self.index)

    def _pad(self, patch, X, Y, H, W):
        ph, pw = patch.shape[:2]
        if (ph, pw) == (self.patch, self.patch):
            return patch
        pl = max(self.ps - X, 0); pr = max(X + self.ps - W, 0)
        pt = max(self.ps - Y, 0); pb = max(Y + self.ps - H, 0)
        patch = np.pad(patch, ((pt, pb), (pl, pr), (0, 0)), mode="constant")
        return patch[:self.patch, :self.patch]

    def __getitem__(self, k):
        ii, X, Y = self.index[k]
        img = self.imgs[ii]; H, W = img.shape[:2]
        b = int(np.clip(Y - self.ps, 0, H)); t = int(np.clip(Y + self.ps, 0, H))
        l = int(np.clip(X - self.ps, 0, W)); r = int(np.clip(X + self.ps, 0, W))
        patch = self._pad(img[b:t, l:r], X, Y, H, W)
        p = self.tf_norm(self.tf_all(patch))
        return p, ii, X, Y


class SporaROSIEWrapper(SporaModelWrapper):
    """In-process, batched. ConvNeXt-Small patch regressor (50 outputs) + Gaussian
    overlap-add blending, faithful to ROSIE's evaluate.py with postprocessing OFF
    (the notebook's default). Only markers requested via `target_markers` are accumulated,
    so memory stays small.

    `predict_markers_from_he_batch` runs one DataLoader over ALL patches of ALL input tiles
    (rather than looping tile-by-tile), which is the main speedup over a naive per-tile call
    to `predict_markers_from_he`.
    """
    channels = ROSIE_CHANNELS
    PATCH = 128

    def __init__(self,
                model_name: str,
                half: bool = True,
                grid_stride: int = 8,
                batch_size: int = 256,
                num_workers: int = 8,
                exclude_bg: bool = False,
                ):
        super().__init__(model_name)
        import torchvision.models as tvm
        from huggingface_hub import snapshot_download

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = bool(half) and self.device == "cuda"
        self.grid_stride = grid_stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exclude_bg = exclude_bg

        d = Path(snapshot_download(repo_id="ericwu09/ROSIE", tqdm_class=tqdm))
        weights = d / "best_model_single.pth"
        m = tvm.convnext_small(weights=None)  # weights come from the ckpt
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(self.channels))
        sd = torch.load(weights, map_location="cpu", weights_only=False)["model_state_dict"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}  # unwrap DataParallel
        m.load_state_dict(sd)
        m = m.to(self.device).eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
        self.model = m

        self.marker_idx = build_marker_index(self.channels)  # canonical name -> ROSIE channel idx
        self.supported_he_markers = set(self.marker_idx.keys())

    def predict_markers_from_he(self,
                                he_tile: torch.Tensor,
                                target_markers: Optional[Iterable[str]] = None,
                                ) -> Dict[str, torch.Tensor]:
        return self.predict_markers_from_he_batch([he_tile], target_markers=target_markers)[0]

    @torch.no_grad()
    def predict_markers_from_he_batch(self,
                                      he_tiles: List[torch.Tensor],
                                      target_markers: Optional[Iterable[str]] = None,
                                      foot: Optional[int] = None,
                                      ) -> List[Dict[str, torch.Tensor]]:
        keep_canon = sorted(self.supported_he_markers if target_markers is None
                            else (set(target_markers) & self.supported_he_markers))
        if not keep_canon:
            return [{} for _ in he_tiles]
        keep_ch = sorted({self.marker_idx[c] for c in keep_canon})
        canon_by_ch = {self.marker_idx[c]: c for c in keep_canon}

        he_images = [denorm_he_uint8(t) for t in he_tiles]
        foot = foot or 2 * self.grid_stride
        row = {c: i for i, c in enumerate(keep_ch)}
        ds = _ROSIEPatchDataset(he_images, self.PATCH, self.grid_stride, self.exclude_bg)
        if len(ds) == 0:
            return [{} for _ in he_tiles]
        dl = DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                        pin_memory=(self.device == "cuda"))

        hw = {i: img.shape[:2] for i, img in enumerate(he_images)}
        acc = {i: np.zeros((len(keep_ch), h, w), np.float32) for i, (h, w) in hw.items()}
        wgt = {i: np.zeros((h, w), np.float32) for i, (h, w) in hw.items()}

        yy, xx = np.mgrid[0:foot, 0:foot]
        c0 = foot // 2
        wker = np.exp(-((xx - c0) ** 2 + (yy - c0) ** 2) / (2 * (foot / 4) ** 2)).astype(np.float32)
        hf = foot // 2
        keep_t = torch.as_tensor(keep_ch, device=self.device)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half)

        for patches, ii, X, Y in tqdm(dl, desc=f"[ROSIE] {len(ds)} patches"):
            patches = patches.to(self.device, non_blocking=True)
            with amp:
                out = self.model(patches)  # (B, 50)
            out = out.float().index_select(1, keep_t).cpu().numpy()  # (B, K)
            ii = ii.numpy(); X = X.numpy(); Y = Y.numpy()
            for j in range(out.shape[0]):
                i = int(ii[j]); x = int(X[j]); y = int(Y[j])
                h, w = hw[i]
                t0 = max(y - hf, 0); b0 = min(y + hf, h)
                l0 = max(x - hf, 0); r0 = min(x + hf, w)
                kt = t0 - (y - hf); kl = l0 - (x - hf)
                wk = wker[kt:kt + (b0 - t0), kl:kl + (r0 - l0)]
                acc[i][:, t0:b0, l0:r0] += out[j][:, None, None] * wk[None]
                wgt[i][t0:b0, l0:r0] += wk

        results = []
        for i in range(len(he_images)):
            a = acc[i] / np.maximum(wgt[i], 1e-8)[None]
            results.append({canon_by_ch[ch]: torch.from_numpy(a[row[ch]]) for ch in keep_ch})
        return results
