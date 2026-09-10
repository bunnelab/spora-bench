import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from spora_bench.utils.setup_utils import clone_git_repo
from spora_bench.utils.virtual_staining_utils import build_marker_index, denorm_he_uint8
from spora_bench.wrapper import SporaModelWrapper

# Fallback marker list, matching the released HistoPlexer demo config.
HISTOP_CHANNELS_FALLBACK = [
    "CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10",
]


class SporaHistoPlexerWrapper(SporaModelWrapper):

    def __init__(self,
                model_name: str,
                work_dir: str,
                binarize: bool = False,
                ):
        super().__init__(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.binarize = binarize

        d = Path(work_dir) / "HistoPlexer"
        clone_git_repo("https://github.com/ratschlab/HistoPlexer.git", d)
        sys.path.insert(0, str(d))
        from src.models.generator import unet_translator

        cfg_path = d / "demo" / "model" / "config.json"
        ckpt = d / "demo" / "model" / "pytorch_model.pt"
        with open(cfg_path) as f:
            cfg = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        self.channels = list(getattr(cfg, "markers", HISTOP_CHANNELS_FALLBACK))
        self.model = unet_translator(
            input_nc=cfg.input_nc, output_nc=cfg.output_nc,
            use_high_res=cfg.use_high_res, use_multiscale=cfg.use_multiscale,
            ngf=cfg.ngf, depth=cfg.depth,
            encoder_padding=cfg.encoder_padding, decoder_padding=cfg.decoder_padding,
            device="cpu", extra_feature_size=cfg.fm_feature_size)
        sd = torch.load(ckpt)
        self.model.load_state_dict(sd["trans_ema_state_dict"])
        self.model.to(self.device).eval()

        self.marker_idx = build_marker_index(self.channels)
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

        he_uint8 = denorm_he_uint8(he_tile)
        x = he_uint8.transpose(2, 0, 1).astype(np.float32)
        x = np.ascontiguousarray(x)
        x = (x // 255) if self.binarize else (x / 255.0)
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        _, _, pred = self.model(x)
        pred = pred.squeeze(0).cpu() 

        return {c: pred[self.marker_idx[c]] for c in keep_canon}
