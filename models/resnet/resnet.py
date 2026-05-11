from typing import Tuple

import torch
from einops import rearrange, repeat
from loguru import logger
from spora_io import MultiplexImagingDataset
from spora_io.datasets._types import Tissue
import numpy as np

from spora_bench.wrapper import SporaModelWrapper
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import MiniBatchSparsePCA

class SporaResNetWrapper(SporaModelWrapper):
    def __init__(self, 
                 model_name: str = "resnet50",
                 pca_dim_channel: int = 9,
                 tile_size: int = 224,
                 ):
        super().__init__(model_name)
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.cuda()
        self.model_dim = self.model.fc.in_features
        self.tile_size = tile_size
        self.pca_dim_channel = pca_dim_channel
        self.dim_per_channel = self.model.fc.out_features

    @torch.inference_mode()
    def embed_tile(self,
                   tissue: Tissue,
                   ) -> torch.Tensor:
        # 1. Load data
        tissue_image = tissue.image.cuda() # C x H x W

        # 2. Embed tile using the model
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
           embeddings = self.model(tissue_image)
        return embeddings
    
    
    @torch.inference_mode()
    def embed_tissue(self,
                     dataset: MultiplexImagingDataset,
                     tissue_id: str,
                     tissue_threshold: float = 0.3
                     ) -> torch.Tensor:
        # 0. Select the cut of channels across all tissues for dimensionality reasons.
        channels_per_tissue = dataset.image_channel_map
        valid_channels = channels_per_tissue.columns[channels_per_tissue.all(axis=0)]

        # 1. Load data
        tissue = dataset.get_tissue(tissue_id, kind="complete", preprocess=True, image_mode="CHW")
        tissue_image = tissue.image
        channel_names = tissue.channel_names
        channel_mask = np.isin(channel_names, valid_channels)
        tissue_image = tissue_image[channel_mask]
        C, H, W = tissue_image.shape
        tissue_mask = dataset.get_tissue_mask(tissue_id).mask
        
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
        for i in x_crops:
            for j in y_crops:
                tile_mask = tissue_mask[i:i+self.tile_size, j:j+self.tile_size]
                if tile_mask.mean() > tissue_threshold:
                    tile = tissue_image[:, i:i+self.tile_size, j:j+self.tile_size]
                    tile = repeat(tile, "c h w -> c rgb h w", rgb=3)  # Repeat channels to get 3 channels for resnet input (rgb)
                    embedding = self.model(tile.cuda())
                    embedding = rearrange(embedding, "c d -> (c d)")
                    embedding_bag.append(embedding.cpu())
                        
        if len(embedding_bag) == 0:
            logger.warning(f"No tiles in tissue {tissue_id} passed the tissue threshold. Returning an empty embedding.")
            return torch.tensor([])
        embedding_bag = torch.vstack(embedding_bag)
        return embedding_bag
    
    def postprocess_tile_embeddings(self,
                                      tissue_embeddings: torch.Tensor):
        sparse_pca = MiniBatchSparsePCA(n_components=self.pca_dim_channel, batch_size=500, random_state=42)
        reduced_embedding = sparse_pca.fit_transform(tissue_embeddings.cpu().numpy())
        return torch.from_numpy(reduced_embedding).float()