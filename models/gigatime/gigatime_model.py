import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from spora_bench.utils.virtual_staining_utils import build_marker_index
from spora_bench.wrapper import SporaModelWrapper

# GigaTIME: 23 channels; indices 1,2 (TRITC, Cy5) are background, excluded from analysis.
GIGA_CHANNELS = [
    "DAPI", "TRITC", "Cy5",
    "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16", "CD11c",
    "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
    "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin",
]
GIGA_BG = {1, 2}


class SporaGigaTIMEWrapper(SporaModelWrapper):
    """In-process GigaTIME UNet++. Feeds imagenet-normalized H&E directly (matches the
    official testing notebook's albumentations Normalize() == imagenet mean/std, which is
    exactly what spora_io's 'he' modality already outputs). GigaTIME emits per-pixel sigmoid
    presence masks, NOT intensities; we use the continuous sigmoid prob as the comparable
    surrogate (documented)."""
    channels = GIGA_CHANNELS

    def __init__(self,
                model_name: str,
                work_dir: str,
                ):
        super().__init__(model_name)
        import subprocess
        from huggingface_hub import snapshot_download

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        d = Path(work_dir) / "gigatime_code"
        if not d.exists():
            subprocess.run(["git", "clone",
                            "https://github.com/prov-gigatime/GigaTIME.git", str(d)], check=True)
        sys.path.insert(0, str(d / "scripts"))
        import archs

        self.model = archs.gigatime(num_classes=23, input_channels=3)
        wdir = Path(snapshot_download(repo_id="prov-gigatime/GigaTIME"))
        sd = torch.load(wdir / "model.pth", map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

        # canonical name -> GigaTIME channel idx, excluding background channels
        self.marker_idx = {c: i for c, i in build_marker_index(self.channels).items() if i not in GIGA_BG}
        self.supported_he_markers = set(self.marker_idx.keys())

    @torch.no_grad()
    def predict_markers_from_he(self,
                                he_tile: torch.Tensor,
                                target_markers: Optional[Iterable[str]] = None,
                                ) -> Dict[str, torch.Tensor]:
        keep_canon = (self.supported_he_markers if target_markers is None
                     else (set(target_markers) & self.supported_he_markers))
        if not keep_canon:
            return {}

        x = he_tile.unsqueeze(0).to(self.device)
        if x.shape[-2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        logits = self.model(x)
        prob = torch.sigmoid(logits).squeeze(0).cpu()  # (23, H, W), continuous surrogate for the binary mask

        return {c: prob[self.marker_idx[c]] for c in keep_canon}
