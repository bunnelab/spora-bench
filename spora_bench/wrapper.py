import os
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple, Optional
import torch
from spora_io.datasets import MultiplexImagingDataset, MultiplexTissue, CellMask


class SporaModelWrapper(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name


    def compute_cell_tokens(self,
                            dataset: MultiplexImagingDataset,
                            tissue_id: str,
                            ) -> torch.Tensor:
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
                   ) -> torch.Tensor:
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
                      ) -> torch.Tensor:
        """
        Predict a target marker given multipelxed input image.
        Args:
            tissue (MultiplexTissue): The input multiplexed tissue. Shape: (C, H, W)
            target_channel_name (str): The name of the target channel to be inpainted.
            target_uniprot_id (str): The uniprot ID of the target channel to be inpainted.
        Returns:
            torch.Tensor: The predicted single-marker image. Shape: (1, H, W)
        Raises:
            MarkerNotSupportedError: If the target uniprot ID is not supported by the model.
        """
        raise NotImplementedError("Inpainting is not implemented for this model.")


    def predict_instance_segmentation(self,
                                      tissue: MultiplexTissue,
                                      ) -> torch.Tensor:
        """
        Predicts the instance segmentation mask for the given tissue. Cell instances are enumerated with unique integers starting from 1.
        Args:
            tissue (MultiplexTissue): The input multiplexed tissue. Shape: (C, H, W)
        Returns:
            torch.Tensor: The predicted instance segmentation mask. Shape: (H, W)
        """
        raise NotImplementedError("Instance segmentation is not implemented for this model.")
    
    def predict_cell_types(self,
                           tissue: MultiplexTissue,
                           mask: CellMask,
                           exclude_classes: Optional[List[str]] = None,
                           ) -> torch.Tensor:

        
        """
        Predicts the cell types for the given tissue. Cell types are enumerated with unique integers starting from 1.
        Args:
            tissue (MultiplexTissue): The input multiplexed tissue. Shape: (C, H, W)
            mask (CellMask): The input cell mask. Shape: (H, W)
            exclude_classes (List[str]): A list of cell types to exclude.
        Returns:
            torch.Tensor: The predicted cell type class ids for each cell instance in the mask. Shape: (num_cells,) where num_cells is the number of unique cells in the mask.
        """
        raise NotImplementedError("Cell type prediction is not implemented for this model.")
    

    def predict_markers_from_he(self,
                                he_tile: torch.Tensor,
                                target_markers: Optional[Iterable[str]] = None,
                                ) -> Dict[str, torch.Tensor]:
        """
        Predict marker maps from an H&E tile alone (no multiplex input required).
        Unlike `predict_marker`, this is for models that translate H&E directly into
        multiplex marker channels (e.g. ROSIE, HistoPlexer, GigaTIME), rather than
        inpainting a dropped multiplex channel from the remaining multiplex channels.

        Implementations should canonicalize their native channel vocabulary into the
        shared marker tokens used across models/datasets (see
        `spora_bench.utils.virtual_staining_utils.canon`/`build_marker_index`), and are
        expected to expose the canonical markers they support via a `supported_he_markers`
        attribute (a set of canonical marker name strings) set in `__init__`, so callers can
        compute the scored marker set without running inference.

        Args:
            he_tile (torch.Tensor): Imagenet-normalized H&E tile. Shape: (3, H, W)
            target_markers (Optional[Iterable[str]]): Canonical marker names to restrict/optimize
                prediction for. Implementations may ignore this and return every marker they support.
        Returns:
            Dict[str, torch.Tensor]: Canonical marker name -> predicted map. Shape of each value: (H, W)
        """
        raise NotImplementedError("H&E-to-marker prediction is not implemented for this model.")


    def predict_markers_from_he_batch(self,
                                      he_tiles: List[torch.Tensor],
                                      target_markers: Optional[Iterable[str]] = None,
                                      ) -> List[Dict[str, torch.Tensor]]:
        """
        Predict marker maps for a batch of H&E tiles. Default implementation loops over
        `predict_markers_from_he` one tile at a time; override for models that benefit from
        batching inference across tiles (e.g. patch-based models like ROSIE).
        Args:
            he_tiles (List[torch.Tensor]): Imagenet-normalized H&E tiles, each of shape (3, H, W).
            target_markers (Optional[Iterable[str]]): See `predict_markers_from_he`.
        Returns:
            List[Dict[str, torch.Tensor]]: One canonical-marker-name -> predicted map dict per input tile.
        """
        return [self.predict_markers_from_he(t, target_markers=target_markers) for t in he_tiles]
        



class MarkerNotSupportedError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
