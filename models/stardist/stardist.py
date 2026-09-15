from spora_bench.wrapper import SporaModelWrapper
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import torch
import numpy as np


NUCLEAR_CHANNELS = [
    "P68431", # Histone H3 (and other nuclear markers)
]

class SporaStarDistWrapper(SporaModelWrapper):

    def __init__(self,
                 model_name: str,
                 ):
        super().__init__(model_name)
        self.model = StarDist2D.from_pretrained("2D_versatile_fluo")

    def predict_instance_segmentation(self, image):
        """
        Predicts instance segmentation for the given image using the StarDist model.
        Args:
            image: The input image to be segmented.
        Returns:
            predicted_mask: The predicted instance segmentation mask.
        """
        tissue_image = image.image
        markers = image.uniprot_ids
        nuclear_chs = [i for i, marker in enumerate(markers) if marker in NUCLEAR_CHANNELS][:1]
        if len(nuclear_chs) == 0:
            raise ValueError("No nuclear channels found in the image.")
        nuc_img = tissue_image[nuclear_chs]
        nuc_img = np.array(nuc_img)
        nuc_img = normalize(nuc_img).transpose(1, 2, 0)  # Normalize and transpose to (H, W, C)
        
        # Predict instance segmentation using StarDist
        labels, _ = self.model.predict_instances(nuc_img)
        return torch.from_numpy(labels)