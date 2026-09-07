"""Shared utilities for H&E-to-multiplex virtual staining benchmarks.

Marker canonicalization: every model has its own channel vocabulary (e.g. ROSIE's
"PanCK" vs. a dataset's "cytokeratin") and every dataset has its own channel naming
convention. `canon()` maps any raw channel/marker name to a single canonical token so
cross-model / cross-dataset marker intersection works, independent of naming source.
"""
import re

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Map every alias we might see (dataset channel_names + model vocabularies) to a single
# canonical token so cross-model / dataset intersection works.
ALIAS_GROUPS = {
    "DAPI": ["dapi", "hoechst", "dna", "nuclei"],
    "CD3": ["cd3", "cd3e", "cd3d"],
    "CD4": ["cd4"],
    "CD8": ["cd8", "cd8a"],
    "CD11C": ["cd11c", "itgax"],
    "CD14": ["cd14"],
    "CD16": ["cd16", "fcgr3a"],
    "CD20": ["cd20", "ms4a1"],
    "CD21": ["cd21"],
    "CD31": ["cd31", "pecam1", "pecam-1"],
    "CD34": ["cd34"],
    "CD38": ["cd38"],
    "CD39": ["cd39"],
    "CD40": ["cd40"],
    "CD44": ["cd44"],
    "CD45": ["cd45", "ptprc"],
    "CD45RO": ["cd45ro"],
    "CD66": ["cd66", "cd66b", "ceacam"],
    "CD68": ["cd68"],
    "CD79A": ["cd79a"],
    "CD138": ["cd138", "syndecan1", "sdc1"],
    "CD141": ["cd141", "thbd"],
    "CD163": ["cd163"],
    "FOXP3": ["foxp3"],
    "PD1": ["pd1", "pd-1", "pdcd1"],
    "PDL1": ["pdl1", "pd-l1", "cd274"],
    "KI67": ["ki67", "ki-67", "mki67"],
    "PANCK": ["panck", "pan-ck", "pan_ck", "ck", "cytokeratin", "keratin", "keratin8/18", "panckeratin"],
    "ECAD": ["ecad", "e-cadherin", "ecadherin", "cdh1"],
    "EPCAM": ["epcam"],
    "SMA": ["sma", "asma", "a-sma", "acta2", "alphasma"],
    "VIMENTIN": ["vimentin", "vim"],
    "PODOPLANIN": ["podoplanin", "pdpn"],
    "HLADR": ["hladr", "hla-dr"],
    "HLAABC": ["hlaabc", "hla-abc", "hlaa", "hla-a", "hlae", "hla-e"],  # keep MHC-I family together
    "GRANZYMEB": ["granzymeb", "gzmb"],
    "BCL2": ["bcl2"],
    "LAG3": ["lag3"],
    "ICOS": ["icos"],
    "GATA3": ["gata3"],
    "GAL3": ["gal3", "galectin3", "lgals3"],
    "TIGIT": ["tigit"],
    "VISTA": ["vista", "vsir"],
    "MPO": ["mpo"],
    "PCNA": ["pcna"],
    "ATM": ["atm"],
    "TP63": ["tp63", "p63"],
    "IFNG": ["ifng", "ifn-g", "ifngamma"],
    "IDO1": ["ido1", "ido"],
    "COLLAGENIV": ["collageniv", "collagen-iv", "col4"],
    "MELANA": ["melana", "melan-a", "mart1", "mlana"],
    "S100": ["s100"],
    "GP100": ["gp100", "pmel", "hmb45"],
    "SOX10": ["sox10"],
    "TBET": ["tbet", "t-bet", "tbx21"],
    "TRYPTASE": ["tryptase", "tpsab1"],
    "TRANSGELIN": ["transgelin", "tagln", "sm22"],
    # background / structural channels we never score
    "_BG_TRITC": ["tritc"],
    "_BG_CY5": ["cy5"],
    "_BG_ACTIN": ["actin-d", "actind", "actin"],
    "_BG_CASP3": ["caspase3-d", "caspase3", "casp3"],
    "_BG_PHH3": ["phh3-b", "phh3"],
}
_TOKEN2CANON = {}
for _canon, _aliases in ALIAS_GROUPS.items():
    for _alias in _aliases:
        _TOKEN2CANON[_alias] = _canon


def canon(name: str):
    """Map a raw channel/marker name to its canonical token, or None if unknown/background."""
    key = re.sub(r"[^a-z0-9/]", "", str(name).lower())
    c = _TOKEN2CANON.get(key)
    if c is None:
        c = _TOKEN2CANON.get(key.split("/")[0])
    if c is None or c.startswith("_BG"):
        return None
    return c


def build_marker_index(names):
    """canon_token -> first channel index in `names` (skips unknown/background)."""
    out = {}
    for i, n in enumerate(names):
        c = canon(n)
        if c is not None and c not in out:
            out[c] = i
    return out


def denorm_he_uint8(he_chw: torch.Tensor) -> np.ndarray:
    """imagenet-normalized (3,H,W) tensor -> uint8 HWC RGB in [0,255]."""
    x = he_chw.detach().cpu().numpy().astype(np.float32)
    x = x * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (1, 2, 0))  # HWC
    return (x * 255.0).round().astype(np.uint8)


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


try:
    from skimage.metrics import structural_similarity as _ssim

    def ssim(pred01, gt01):
        return float(_ssim(gt01, pred01, data_range=1.0))
except Exception:  # self-contained gaussian SSIM fallback
    def ssim(pred01, gt01):
        import scipy.ndimage as ndi
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu1 = ndi.gaussian_filter(gt01, 1.5)
        mu2 = ndi.gaussian_filter(pred01, 1.5)
        s1 = ndi.gaussian_filter(gt01 * gt01, 1.5) - mu1 * mu1
        s2 = ndi.gaussian_filter(pred01 * pred01, 1.5) - mu2 * mu2
        s12 = ndi.gaussian_filter(gt01 * pred01, 1.5) - mu1 * mu2
        m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2) + 1e-12)
        return float(m.mean())


def crop_metrics(pred2d: np.ndarray, gt2d: np.ndarray):
    """pred2d, gt2d: same-shape float maps. Returns (pearson, mse_norm, ssim)."""
    p = pred2d.astype(np.float32).ravel()
    g = gt2d.astype(np.float32).ravel()
    if p.std() < 1e-8 or g.std() < 1e-8:
        pear = np.nan
    else:
        pear = float(np.corrcoef(p, g)[0, 1])
    pn, gn = _minmax(pred2d.astype(np.float32)), _minmax(gt2d.astype(np.float32))
    mse = float(np.mean((pn - gn) ** 2))
    ss = ssim(pn, gn)
    return pear, mse, ss


def resize2d(x: np.ndarray, hw) -> np.ndarray:
    if x.shape == tuple(hw):
        return x
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    t = F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()
