from typing import Optional, Tuple

import torch
from loguru import logger
from einops import rearrange
from safetensors.torch import load_file
from spora_io import MultiplexImagingDataset
from spora_io.datasets._types import MultiplexTissue
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.cell_tokens import Tuple, compute_cell_tokens
from virtues.utils.utils import (load_marker_embedding_dict,
                                 load_marker_embeddings)

from spora_bench.wrapper import SporaModelWrapper


class SporaVirTuesWrapper(SporaModelWrapper):
    def __init__(self, 
                 model_name: str,
                 checkpoint_path: str,
                 marker_embeddings_dir: str,
                 patch_size: int,
                 model_dim: int,
                 feedforward_dim: int,
                 encoder_pattern: str,
                 num_encoder_heads: int,
                 decoder_pattern: str,
                 num_decoder_heads: int,
                 num_decoder_hidden_layers: int,
                 positional_embedding_type: str,
                 dropout: float,
                 group_layers: bool,
                 norm_after_encoder_decoder: bool,
                 tile_size: int = 128,
                 ):
        super().__init__(model_name)

        marker_embeddings = load_marker_embeddings(marker_embeddings_dir)
        self.marker_embedding_dict = load_marker_embedding_dict(marker_embeddings_dir)
        self.patch_size = patch_size
        self.tile_size = tile_size

        self.model = MultiplexVirtues(
            prior_bias_embeddings=marker_embeddings,
            prior_bias_embedding_type='esm',
            prior_bias_embedding_fusion_type='add',
            patch_size=patch_size,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            encoder_pattern=encoder_pattern,
            num_encoder_heads=num_encoder_heads,
            decoder_pattern=decoder_pattern,
            num_decoder_heads=num_decoder_heads,
            num_hidden_layers=num_decoder_hidden_layers,
            positional_embedding_type=positional_embedding_type,
            dropout=dropout,
            group_layers=group_layers,
            norm_after_encoder_decoder=norm_after_encoder_decoder,
            verbose=False
        )

        checkpoint = load_file(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.cuda()


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