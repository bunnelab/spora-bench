import torch
import numpy as np

import pandas as pd
from omegaconf import OmegaConf

from Eva.utils import load_from_hf

from spora_bench.wrapper import SporaModelWrapper, MarkerNotSupportedError
from spora_io.datasets._types import MultiplexTissue
from loguru import logger


class SporaEvaWrapper(SporaModelWrapper):

    def __init__(self,
                 model_name: str,
                 eva_config_path: str,
                 uniprot_mapping_path: str,
                 tile_size: int = 224,
                 patch_size: int = 8,
                 ):
        super().__init__(model_name=model_name)

        self.uniprot_to_gene = pd.read_parquet(uniprot_mapping_path).set_index('uniprot_id').to_dict()['gene_name']

        self.tile_size = tile_size
        self.patch_size = patch_size

        eva_config = OmegaConf.load(eva_config_path)

        self.model = load_from_hf(
            repo_id="yandrewl/Eva",
            conf=eva_config,
            device='cuda'
        )

        self.allowed_marker_names = []
        self.allowed_marker_names.extend(list(self.model.model.marker_embed.unknown_marker_embeddings.keys()))
        self.allowed_marker_names.extend(list(self.model.model.marker_embed.genept_embeddings.keys()))
        logger.info(f"Eva model loaded with {len(self.allowed_marker_names)} allowed marker names.")

    @torch.inference_mode()
    def predict_marker(self,
                      tissue:  MultiplexTissue,
                      target_channel_name: str,
                      target_uniprot_id: str,
                      ):

        # Check if the target uniprot ID is in the model's uniprot_to_gene mapping
        if target_uniprot_id not in self.uniprot_to_gene:
            raise MarkerNotSupportedError("Target uniprot ID not found in the model's uniprot_to_gene mapping.")
        if self.uniprot_to_gene[target_uniprot_id] not in self.allowed_marker_names:
            raise MarkerNotSupportedError(f"Target marker {self.uniprot_to_gene[target_uniprot_id]} is not supported by Eva.")

        x = tissue.image # (C, H, W)
        uniprot_ids = tissue.uniprot_ids # np.array
        marker_names = np.array([self.uniprot_to_gene.get(uniprot_id, '-1') for uniprot_id in uniprot_ids])

        # Filter out channels that are not supported by Eva
        keep_mask = np.isin(uniprot_ids, list(self.uniprot_to_gene.keys())) & np.isin(marker_names, self.allowed_marker_names)
        if not np.all(keep_mask):
            logger.warning(f"Some markers are not supported by Eva and will be ignored: Uniprot IDs: {uniprot_ids[~keep_mask]} Marker names: {marker_names[~keep_mask]}")

        # Apply filter
        x = x[keep_mask, :, :]
        uniprot_ids = uniprot_ids[keep_mask].tolist()

        # Pad with empty channel for the target marker
        tgt = torch.zeros_like(x[0:1, :, :]) # (1, H, W)
        x = torch.cat([tgt, x], dim=0) # (C+1, H, W)
        uniprot_ids = [target_uniprot_id] + uniprot_ids
        marker_names = [self.uniprot_to_gene[u] for u in uniprot_ids]

        # Prepare mask
        C, H, W = x.shape
        mask = torch.zeros((C, (self.tile_size//self.patch_size)**2), dtype=torch.bool)
        mask[0] = True

        x = x.cuda()
        mask = mask.cuda()

        # Prepare sliding window
        stride = self.tile_size // 1
        rcoords = list(range(0, H-self.tile_size+1, stride)) + [H-self.tile_size] if (H - self.tile_size) % stride != 0 else list(range(0, H-self.tile_size+1, stride))
        ccoords = list(range(0, W-self.tile_size+1, stride)) + [W-self.tile_size] if (W - self.tile_size) % stride != 0 else list(range(0, W-self.tile_size+1, stride))
        
        out = torch.zeros((H,W), dtype=torch.float32)
        num_terms = torch.zeros_like(out)

        for row, col in [(row, col) for row in rcoords for col in ccoords]:
            chunk = x[:, row:row+self.tile_size, col:col+self.tile_size] # (C, tile_size, tile_size)
            chunk = chunk.unsqueeze(0) # (1, C, tile_size, tile_size)
            chunk = chunk.permute(0, 2, 3, 1) # (1, tile_size, tile_size, C)

            recon, _, _ = self.model(
                chunk,
                marker_in=[marker_names],
                marker_out=None,
                infer_mask=mask,
                channel_mask=None,
            ) # (1, tile_size, tile_size, C)

            predicted_marker = recon[0, :, :, 0].cpu() # (tile_size, tile_size)
            out[row:row+self.tile_size, col:col+self.tile_size] += predicted_marker
            num_terms[row:row+self.tile_size, col:col+self.tile_size] += 1

        out /= num_terms
        return out.unsqueeze(0) # (1, H, W)
