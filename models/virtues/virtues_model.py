from typing import Optional, Tuple

import torch
from loguru import logger
from einops import rearrange
from safetensors.torch import load_file
from spora_io import MultiplexImagingDataset
from spora_io.datasets._types import MultiplexTissue, CellMask
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.cell_tokens import Tuple, compute_cell_tokens
from virtues.utils.utils import (load_marker_embedding_dict,
                                 load_marker_embeddings)
from virtues.modules.segmentation.unet import VirtuesSegmentationHead
from virtues.utils.segmentation import remove_small_cells, assign_cell_types


from spora_bench.wrapper import SporaModelWrapper


class SporaVirTuesWrapper(SporaModelWrapper):

    def __init__(self, 
                 model_name: str,
                 checkpoint_path: str,
                 marker_embeddings_dir: str,
                 tile_size: int = 128,
                 segmentation_checkpoint_path: Optional[str] = None,
                 num_classes: Optional[int] = 9,
                 ):
        """
        Initializes the SporaVirTuesWrapper.

        Args:
            model_name (str): Name of the model.
            checkpoint_path (str): Path to the model checkpoint.
            marker_embeddings_dir (str): Directory containing marker embeddings.
            tile_size (int, optional): Size of the tiles to be used for embedding. Defaults to 128.
            segmentation_checkpoint_path (Optional[str], optional): Path to the segmentation model checkpoint. Defaults to None.
            num_classes (Optional[int], optional): Number of classes for semantic segmentation. Defaults to 9 for public  segmentation weights.
        """
        super().__init__(model_name)

        marker_embeddings = load_marker_embeddings(marker_embeddings_dir)
        self.marker_embedding_dict = load_marker_embedding_dict(marker_embeddings_dir)
        self.tile_size = tile_size

        self.model = MultiplexVirtues(
            prior_bias_embeddings=marker_embeddings,
        )

        self.patch_size = self.model.encoder.patch_size

        checkpoint = load_file(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.cuda()

        self.segmentation_model = None

        if segmentation_checkpoint_path is not None:
            segmentation_checkpoint = load_file(segmentation_checkpoint_path)
            self.segmentation_model = VirtuesSegmentationHead(
                virtues_model=self.model,
                dim_out=5,
                num_celltypes=num_classes,
            )
            self.segmentation_model.load_state_dict(segmentation_checkpoint, strict=False)
            self.segmentation_model.eval()
            self.segmentation_model.cuda()

    @torch.inference_mode()
    def compute_cell_tokens(self,
                            dataset: MultiplexImagingDataset,
                            tissue_id: str,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Load data
        tissue = dataset.get_tissue(tissue_id, kind='uniprot_filtered', preprocess=False, image_mode="CHW")
        try:
            segmentation_mask = dataset.get_cell_instance_mask(tissue_id)
            segmentation_mask = segmentation_mask.mask
        except ValueError:
            logger.warning(f"Cell instance mask not found for tissue ID {tissue_id}. Skipping cell token computation.")
            return torch.tensor([]), torch.tensor([])

        if min(tissue.image.shape[1:]) < self.tile_size:
            logger.warning(f"Tissue {tissue_id} is smaller than the tile size. Skipping cell token computation.")
            return torch.tensor([]), torch.tensor([])

        if segmentation_mask.max() == 0:
            logger.warning(f"Segmentation mask for tissue {tissue_id} is empty. Skipping cell token computation.")
            return torch.tensor([]), torch.tensor([])

        assert tissue.image.shape[1:] == segmentation_mask.shape, f"Image and segmentation mask shapes do not match for tissue ID {tissue_id}."
        
        # 2. Pad image and segmentation mask pre-standardization
        pad_size = 120
        x = torch.nn.functional.pad(tissue.image, pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        segmentation_mask = torch.nn.functional.pad(segmentation_mask, pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)


        # 3. Standardize image
        x, refined_mask = dataset.standardizer.apply(x, tissue_id, tissue.measured_mask, tissue.image_loading_mask)
        image_loading_mask, channel_names, uniprot_ids = dataset._refine_channel_metadata(
            tissue.image_loading_mask,
            tissue.channel_names,
            tissue.uniprot_ids,
            refined_mask,
        )

        # 4. Map uniprot IDs to marker embedding indices
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprot_ids], dtype=torch.long)

        # 5. Compute cell tokens
        cell_ids, cell_tokens, _, _ = compute_cell_tokens(self.model, x, marker_indices, segmentation_mask)

        return cell_ids, cell_tokens
    
    @torch.inference_mode()
    def embed_tile(self,
                   tissue: MultiplexTissue,
                   ) -> torch.Tensor:
        # 1. Load data
        tissue_image = tissue.image.cuda()

        # 2. Map uniprot IDs to marker embedding indices
        uniprot_ids = tissue.uniprot_ids
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprot_ids], dtype=torch.long)

        # 3. Embed tile using the model
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            virtues_output = self.model.encoder.forward_list([tissue_image], [marker_indices])
        tile_embedding = virtues_output.patch_summary_tokens[0].cpu()
        tile_embedding = rearrange(tile_embedding, 'h w d -> (h w) d')
        return tile_embedding
    
    @torch.inference_mode()
    def embed_tissue(self,
                     dataset: MultiplexImagingDataset,
                     tissue_id: str,
                     tissue_threshold: float = 0.3
                     ) -> torch.Tensor:
        # 1. Load data
        tissue = dataset.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True, image_mode="CHW")
        tissue_image = tissue.image.cuda()
        tissue_mask = dataset.get_tissue_mask(tissue_id).mask
        uniprot_ids = tissue.uniprot_ids
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprot_ids], dtype=torch.long)
        marker_indices = marker_indices.cuda()
        C, H, W = tissue_image.shape
        
        # 2. Get crop coordinates
        # make sure to cover the entire tissue image with tiles. If the tile size does not perfectly divide the image dimensions, we need to add an additional tile that overlaps with the last tile to cover the remaining area.
        x_crops = list(range(0, H - self.tile_size, self.tile_size))
        y_crops = list(range(0, W - self.tile_size, self.tile_size))
        if len(x_crops) == 0 or len(y_crops) == 0:
            logger.warning(f"Tissue {tissue_id} is smaller than the tile size. Returning an empty embedding.")
            return torch.tensor([])
        
        if x_crops[-1] != H - self.tile_size:
            x_crops.append(H - self.tile_size)
        if y_crops[-1] != W - self.tile_size:
            y_crops.append(W - self.tile_size)

        embedding_bag = []
        with torch.amp.autocast("cuda", dtype=torch.float16):
            for i in x_crops:
                for j in y_crops:
                    tile_mask = tissue_mask[i:i+self.tile_size, j:j+self.tile_size]
                    if tile_mask.mean() > tissue_threshold:  # Only embed tiles that contain more than the specified threshold of tissue
                        tile = tissue_image[:, i:i+self.tile_size, j:j+self.tile_size]
                        virtues_output = self.model.encoder.forward_list([tile], [marker_indices])
                        patch_token_features = rearrange(virtues_output.patch_summary_tokens[0], "h w d -> (h w) d")
                        embedding_bag.append(patch_token_features.cpu())
        if len(embedding_bag) == 0:
            logger.warning(f"No tiles in tissue {tissue_id} passed the tissue threshold. Returning an empty embedding.")
            return torch.tensor([])
        embedding_bag = torch.cat(embedding_bag, dim=0)
        return embedding_bag
    
           
    @torch.inference_mode()
    def predict_marker(self,
                      tissue:  MultiplexTissue,
                      target_channel_name: str,
                      target_uniprot_id: str,
                      ):

        x = tissue.image # (C, H, W)
        tgt = torch.zeros_like(x[0:1]) # (1, H, W)
        x = torch.concat([tgt, x,], dim=0) # (C+1, H, W)

        uniprot_ids = [target_uniprot_id,] + tissue.uniprot_ids.tolist()
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprot_ids], dtype=torch.long)

        C, H, W = x.shape
        mask = torch.zeros((C, self.tile_size//self.patch_size, self.tile_size//self.patch_size), dtype=torch.bool)
        mask[0] = True

        x = x.cuda()
        marker_indices = marker_indices.cuda()
        mask = mask.cuda()

        stride = self.tile_size // 1
        rcoords = list(range(0, H-self.tile_size+1, stride)) + [H-self.tile_size] if (H - self.tile_size) % stride != 0 else list(range(0, H-self.tile_size+1, stride))
        ccoords = list(range(0, W-self.tile_size+1, stride)) + [W-self.tile_size] if (W - self.tile_size) % stride != 0 else list(range(0, W-self.tile_size+1, stride))

        out = torch.zeros((H, W), dtype=torch.float32)
        num_terms = torch.zeros_like(out)

        for row, col in [(row, col) for row in rcoords for col in ccoords]:
            chunk = x[:, row:row+self.tile_size, col:col+self.tile_size]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model.forward([chunk], [marker_indices], [mask])
            predicted_marker = output.decoded_multiplex[0][0].cpu() # (H,W)
            out[row:row+self.tile_size, col:col+self.tile_size] += predicted_marker
            num_terms[row:row+self.tile_size, col:col+self.tile_size] += 1
        
        out /= num_terms
        return out.unsqueeze(0) # (1, H, W)


    @torch.inference_mode()
    def predict_instance_segmentation(
                                    self,
                                    tissue: MultiplexTissue,
                                    ) -> torch.Tensor:
        """
        Predicts the instance segmentation mask for the given tissue.
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model is not loaded. Please provide a segmentation checkpoint path during initialization.")

        x = tissue.image
        uniprot_ids = tissue.uniprot_ids
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprot_ids], dtype=torch.long)
        marker_indices = marker_indices.cuda()

        pred_instance, _ = self.segmentation_model.segment_tissue(x.cuda(), marker_indices, tile_size=128, overlap=32, batch_size=4)

        pred_instance = pred_instance.cpu()
        pred_instance = remove_small_cells(pred_instance, min_cell_size=15) # returns np.array
        pred_instance = torch.from_numpy(pred_instance)

        return pred_instance

    @torch.inference_mode()
    def predict_cell_types(self,
                           tissue: MultiplexTissue,
                           mask: CellMask,
                           excluded_classes: Optional[list] = None,
                           ):
        """
        Predicts the cell types for the given tissue and instance segmentation mask.
        Args:
            tissue (MultiplexTissue): The input multiplexed tissue. Shape: (C, H, W)
            mask (CellMask): The input cell mask. Shape: (H, W)
            excluded_classes (list, optional): List of class indices to exclude from prediction.
        Returns:
            torch.Tensor: The predicted cell type class ids for each cell instance in the mask. Shape: (num_cells,) where num_cells is the number of unique cells in the mask.
        Raises
        """
        if self.segmentation_model is None:
                    raise ValueError("Segmentation model is not loaded. Please provide a segmentation checkpoint path during initialization.")
        
        x = tissue.image
        uniprot_ids = tissue.uniprot_ids
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprot_ids], dtype=torch.long)
        marker_indices = marker_indices.cuda()

        _, semantic_logits = self.segmentation_model.segment_tissue(x.cuda(), marker_indices, tile_size=128, overlap=32, batch_size=4)
        if excluded_classes is not None:
            semantic_logits[excluded_classes, :, :] = float('-inf')
        semantic_pred = torch.argmax(semantic_logits, dim=0).cpu() # (H, W)

        pred_y = self._majority_vote(mask.mask, semantic_pred) # (num_cells,)
        return pred_y

    
    def _majority_vote(self, instance_mask, semantic_mask) -> torch.Tensor:
        """
        Assigns a cell type to each cell instance in the instance mask based on the majority vote of the semantic mask.
          0 encodes background in the instance mask. The instance mask might not be contiguous. Output order should should follow the order of sorted unique cell IDs in the instance mask.
        Args:
            instance_mask (torch.Tensor): The instance segmentation mask. Shape: (H, W)
            semantic_mask (torch.Tensor): The semantic segmentation mask. Shape: (H, W)
        Returns:
            torch.Tensor: The predicted cell type class ids for each cell instance in the mask. Shape: (num_cells,) where num_cells is the number of unique cells in the mask.
        """
        inst = instance_mask.reshape(-1)
        sem = semantic_mask.reshape(-1).long()

        # drop background pixels of the instance mask
        fg = inst != 0
        inst, sem = inst[fg], sem[fg]

        # sorted unique cell IDs -> contiguous row indices 0..n_cells-1
        cell_ids, rows = torch.unique(inst, return_inverse=True)
        n_cells = cell_ids.numel()
        if n_cells == 0:
            return torch.empty(0, dtype=torch.long, device=instance_mask.device)

        n_classes = int(sem.max()) + 1

        # linearize (cell, class) into a single index, then one bincount
        counts = torch.bincount(
            rows * n_classes + sem,
            minlength=n_cells * n_classes,
        ).reshape(n_cells, n_classes)

        return counts.argmax(dim=1)
