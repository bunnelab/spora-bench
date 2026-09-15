from spora_bench.wrapper import SporaModelWrapper
from cellpose import models as cellpose_models
import numpy as np
import torch

CYTOPLASM_CHANNELS = [
    "P08670", # Vimentin
    "P62736", # SMA
    "P02533", # panCK
    "P04264", # KRT1
    "P26447", # S100
    "P05164", # MPO
    "P10144", # Granzyme B
]

NUCLEAR_CHANNELS = [
    "P68431", # Histone H3 (and other nuclear markers)
]

class SporaCellposeWrapper(SporaModelWrapper):

    def __init__(self,
                    model_name: str,
                    gpu: bool = False,
                 ):
        super().__init__(model_name)
        self.model = cellpose_models.CellposeModel(gpu=gpu)

    def _prepare_image(self, image):
        """
        Prepares the input image for Cellpose model.
        Args:
            image: The input image to be prepared.
        Returns:
            prepared_image: The image after preparation, ready for Cellpose model.
        """
        markers = image.uniprot_ids
        cytoplasm_chs = [i for i, marker in enumerate(markers) if marker in CYTOPLASM_CHANNELS]
        nuclear_chs = [i for i, marker in enumerate(markers) if marker in NUCLEAR_CHANNELS][:1]
        if len(nuclear_chs) == 0:
            nuclear_chs = None
        tissue_image = image.image

        cyto_stack = tissue_image[cytoplasm_chs]
        cyto_p1 = np.percentile(cyto_stack, 1, axis=(1, 2), keepdims=True)
        cyto_p99 = np.percentile(cyto_stack, 99, axis=(1, 2), keepdims=True)
        cyto_scale = np.maximum(cyto_p99 - cyto_p1, 1e-6)
        cyto_stack_norm = np.clip((cyto_stack - cyto_p1) / cyto_scale, 0.0, 1.0)
        cyto_img = cyto_stack_norm.mean(axis=0)

        if nuclear_chs is not None:
            nuc_img = tissue_image[nuclear_chs].mean(axis=0)
            img = np.stack([cyto_img, nuc_img], axis=-1).astype(np.float32)
            channels = [1, 2]
        else:
            img = cyto_img.astype(np.float32)
            channels = [0, 0]

        return img, channels

    def predict_instance_segmentation(self, image):
        img, channels = self._prepare_image(image)
        pred, _, _ = self.model.eval(
            img,
            channels=channels,
            do_3D=False,
        )
        return torch.from_numpy(pred)