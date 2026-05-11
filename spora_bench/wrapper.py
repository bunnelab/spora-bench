import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
from spora_io.datasets import MultiplexImagingDataset, MultiplexTissue


class SporaModelWrapper(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    def compute_cell_tokens(self,
                            dataset: MultiplexImagingDataset,
                            tissue_id: str,
                            ):
        """
        Compute cell tokens for the given dataset and tissue ID.
        Args:
            dataset (MultiplexImagingDataset): The input dataset containing the multiplexed images and segmentation masks.
            tissue_id (str): The ID of the tissue to compute cell tokens for.
        Returns:
            torch.Tensor: The computed cell token tensor. Shape: (N, D)
        """
        raise NotImplementedError("Cell token computation is not implemented for this model.")

    def embed_tile(self,
                   tissue: MultiplexTissue
                   ):
        """
        Embed a tile from the input image.
        Args:
            tissue (MultiplexTissue): The input tissue containing the image and protein IDs.
        Returns:
            torch.Tensor: The embedded tile tensor. Shape: (D,)
        """
        raise NotImplementedError("Tile embedding is not implemented for this model.")
    
    def embed_tissue(self,
                     dataset: MultiplexImagingDataset,
                     tissue_id: str,
                     tissue_threshold: float = 0.3
                     ) -> torch.Tensor:
        """
        Embed the tissue from the input image.
        Args:
            dataset (MultiplexImagingDataset): The input dataset containing the multiplexed images and segmentation masks.
            tissue_id (str): The ID of the tissue to embed.
            tissue_threshold (float): The threshold for considering a tile as containing tissue.
        Returns:
            torch.Tensor: A sequence-shaped embedding of the tissue. Shape: (N,D)
        """
        raise NotImplementedError("Tissue embedding is not implemented for this model.")
    
    def postprocess_tile_embeddings(self, tissue_embedding: torch.Tensor):
        """
        Postprocess the tissue embedding if necessary (e.g., for dimensionality reduction for kronos patient level tasks).
        Args:
            tissue_embedding (torch.Tensor): The raw tissue embedding tensor. Shape: (N, D)
        Returns:
            torch.Tensor: The postprocessed tissue token tensor. Shape: (N, D')
        """
        return tissue_embedding

    def predict_marker(self,
                      tissue:  MultiplexTissue,
                      target_channel_name: str,
                      target_uniprot_id: Optional[str] = None,
                      ):
        """
        Predict a target marker given multipelxed input image.
        Args:
            tissue (MultiplexTissue): The input multiplexed tissue. Shape: (C, H, W)
            target_channel_name (str): The name of the target channel to be inpainted.
            target_uniprot_id (str): The uniprot ID of the target channel to be inpainted.
        Returns:
            torch.Tensor: The predicted single-marker image. Shape: (1, H, W)
        """
        raise NotImplementedError("Inpainting is not implemented for this model.")