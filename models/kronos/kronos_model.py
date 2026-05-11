from typing import Tuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from kronos import create_model_from_pretrained
from loguru import logger
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from tqdm import tqdm
from spora_io.datasets._types import Tissue
from spora_io.datasets.multiplex import MultiplexImagingDataset
from torchvision.transforms.functional import pad, resize
from skimage.transform import rescale

from spora_bench.wrapper import SporaModelWrapper


class SporaKronosWrapper(SporaModelWrapper):
    def __init__(self, 
                 model_name: str,
                 checkpoint_path: str,
                 marker_metadata_path: str,
                 uniprot_mapping: str,
                 tile_size: int = 224,
                 pca_dimensions: int = 256,
                 image_mpp: float = 1,
                 ):
        super().__init__(model_name)
        self.model, self.precision, self.feat_dim = create_model_from_pretrained(
            checkpoint_path=checkpoint_path,
        )
        self.model.eval()
        self.model.cuda()
        self.marker_metadata = pd.read_csv(marker_metadata_path)
        self.uniprot_mapping = pd.read_csv(uniprot_mapping)
        self.marker_metadata.set_index("marker_id", inplace=True)
        self.marker_metadata[["marker_mean", "marker_std"]] = self.marker_metadata[["marker_mean", "marker_std"]].astype(np.float32)
        self.tile_size = tile_size
        self.pca_dimensions = pca_dimensions
        self.image_mpp = image_mpp

    def _preprocess(self,
                    tissue: Tissue
                    ):
        """
        Preprocesses the tissue image and maps protein IDs to marker IDs for Kronos.
        Arguments:
            tissue (Tissue): The input tissue containing the image and protein IDs.
        Returns:
            tissue_image (torch.Tensor): The normalized tissue image.
            marker_ids_kronos (torch.Tensor): The corresponding marker IDs for the tissue image.
        """
        protein_ids = tissue.uniprot_ids
        marker_ids_kronos = []

        for protein_id in protein_ids:
            marker_ids = self.uniprot_mapping.query(f"uniprot_id == '{protein_id}'")["marker_id"].values
            if len(marker_ids) > 0:
                marker_ids_kronos.append(marker_ids[0])
            else:
                marker_ids_kronos.append(np.nan)
        
        marker_ids_kronos = np.array(marker_ids_kronos)
        valid_indices = ~pd.isna(marker_ids_kronos)
        tissue_image = tissue.image[valid_indices]
        marker_ids_kronos = marker_ids_kronos[valid_indices].astype(int)

        # means are needed for background imputation for cell token computation when removing pixel values outside of the cell instance mask
        means, std = torch.tensor(self.marker_metadata.loc[marker_ids_kronos][["marker_mean", "marker_std"]].values.T)
        return tissue_image.float(), torch.tensor(marker_ids_kronos).long(), means


    @torch.inference_mode()
    def embed_tile(self, 
                   tissue: Tissue
                   ) -> torch.Tensor:
        tissue_image, marker_ids_kronos = self._preprocess(tissue)
        _, _, patch_token_features = self.model(x=tissue_image.unsqueeze(0), marker_ids=marker_ids_kronos.unsqueeze(0))
        patch_token_features = rearrange(patch_token_features, "b c h w d -> b (h w) (c d)")
        tile_embedding = patch_token_features.mean(axis=1)[0]
        return tile_embedding

    @torch.inference_mode()
    def embed_tissue(self,
                     dataset: MultiplexImagingDataset,
                     tissue_id: str,
                     tissue_threshold: float = 0.3
                     ) -> torch.Tensor:
        # 0. Select the cut of channels across all tissues for dimensionality reasons.
        channels_per_tissue = dataset.image_channel_map
        valid_channels = channels_per_tissue.columns[channels_per_tissue.all(axis=0)]

        tissue = dataset.get_tissue(tissue_id, kind="qc_filtered", preprocess=True, image_mode="CHW")
        channel_names = tissue.channel_names
        channel_mask = np.isin(channel_names, valid_channels)
        tissue.image = tissue.image[channel_mask]
        tissue.channel_names = tissue.channel_names[channel_mask]
        tissue.uniprot_ids = tissue.uniprot_ids[channel_mask]
        tissue_image, marker_ids_kronos, _ = self._preprocess(tissue)

        # Rescale the tissue to match the 0.5mpp resolution
        tissue_mask = dataset.get_tissue_mask(tissue_id).mask
        tissue_mask = rescale(tissue_mask, self.image_mpp / 0.5, order=0, preserve_range=True).astype(bool)
        tissue_image = rescale(tissue_image.numpy(), (1, self.image_mpp / 0.5, self.image_mpp / 0.5), preserve_range=True)
        tissue_image = torch.from_numpy(tissue_image)
        C, H, W = tissue_image.shape

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


        tissue_image = tissue_image.cuda()
        marker_ids_kronos = marker_ids_kronos.cuda()
        # embed all tiles and concatenate the resulting token features. Apply padding if the tile size does not perfectly divide the image dimensions.
        # we iterate over the image in steps of tile_size
        embedding_bag = []
        with torch.amp.autocast("cuda", dtype=torch.float16):
            for i in x_crops:
                for j in y_crops:
                    tile = tissue_image[:, i:i+self.tile_size, j:j+self.tile_size]
                    tile_mask = tissue_mask[i:i+self.tile_size, j:j+self.tile_size]
                    if tile_mask.mean() > tissue_threshold:  # Only embed tiles that contain more than the specified threshold of tissue
                        _, tile_marker_token, _ = self.model(x=tile.unsqueeze(0), marker_ids=marker_ids_kronos.unsqueeze(0))
                        tile_marker_token = rearrange(tile_marker_token, "b c d -> b (c d)")
                        embedding_bag.append(tile_marker_token[0].cpu())

        if len(embedding_bag) == 0:
            # If no tiles were embedded, return a zero tensor with the appropriate shape
            return torch.tensor([])
        return torch.vstack(embedding_bag)
    
    @torch.inference_mode()
    def compute_cell_tokens(self,
                            dataset: MultiplexImagingDataset,
                            tissue_id: str,
                            batch_size: int = 32,
                            cell_tile_size: int = 64
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 0. Select the cut of channels across all tissues for dimensionality reasons.
        channels_per_tissue = dataset.image_channel_map
        valid_channels = channels_per_tissue.columns[channels_per_tissue.all(axis=0)]

        tissue = dataset.get_tissue(tissue_id, kind="qc_filtered", preprocess=True, image_mode="CHW")
        channel_names = tissue.channel_names
        channel_mask = np.isin(channel_names, valid_channels)
        tissue.image = tissue.image[channel_mask]
        tissue.channel_names = tissue.channel_names[channel_mask]
        tissue.uniprot_ids = tissue.uniprot_ids[channel_mask]

        # pad the tissue image and cell instance mask by half the tile size
        tissue.image = pad(tissue.image, (cell_tile_size//2, cell_tile_size//2))
        tissue_image, marker_ids_kronos, channel_means = self._preprocess(tissue)
        background_values = -channel_means

        try:
            segmentation_mask = dataset.get_cell_instance_mask(tissue_id)
            segmentation_mask = segmentation_mask.mask
        except ValueError:
            logger.warning(f"Cell instance mask not found for tissue ID {tissue_id}. Skipping cell token computation.")
            return torch.tensor([]), torch.tensor([])

        if min(tissue_image.shape[1:]) < self.tile_size:
            logger.warning(f"Tissue {tissue_id} is smaller than the tile size. Skipping cell token computation.")
            return torch.tensor([]), torch.tensor([])

        if segmentation_mask.max() == 0:
            logger.warning(f"Segmentation mask for tissue {tissue_id} is empty. Skipping cell token computation.")
            return torch.tensor([]), torch.tensor([])

        cell_instance_mask = pad(segmentation_mask, (cell_tile_size//2, cell_tile_size//2))
        marker_ids_kronos = marker_ids_kronos.to("cuda", non_blocking=True).unsqueeze(0)
        
        cell_datasets = KronosCellDataset(tissue_image, cell_instance_mask, background_values, tile_size=cell_tile_size, image_mpp=self.image_mpp)
        cell_loader = torch.utils.data.DataLoader(cell_datasets, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
        cell_embeddings = []
        cell_ids = []

        with torch.amp.autocast("cuda", dtype=torch.float16):
            for cell_tiles, ids in cell_loader:
                cell_tiles = cell_tiles.to("cuda")
                _, tile_marker_token, _ = self.model(x=cell_tiles, marker_ids=marker_ids_kronos.expand(cell_tiles.shape[0], -1))
                cell_embedding = rearrange(tile_marker_token, "b c d -> b (c d)")
                cell_embeddings += list(cell_embedding.to("cpu", non_blocking=False))
                cell_ids += list(ids)
        return torch.tensor(cell_ids), torch.vstack(cell_embeddings)
    
    def postprocess_tile_embeddings(self,
                                      tissue_embeddings: torch.Tensor):
        # Apply PCA to reduce dimensionality of tile embeddings
        pca = PCA(n_components=self.pca_dimensions)
        reduced_embeddings = pca.fit_transform(np.array(tissue_embeddings))
        return torch.tensor(reduced_embeddings)


class KronosCellDataset(torch.utils.data.Dataset):
    def __init__(self, tissue_image: torch.Tensor, cell_instance_mask: torch.Tensor, background_values: torch.Tensor, tile_size: int = 64, image_mpp: float = 1):
        self.tile_size = tile_size
        cell_instances = torch.unique(cell_instance_mask)
        self.cell_instances = cell_instances[cell_instances != 0]  # Exclude background
        self.tissue_image = tissue_image
        self.cell_instance_mask = cell_instance_mask
        self.background_values = background_values
        self.image_mpp = image_mpp

    def __len__(self):
        return len(self.cell_instances)

    def __getitem__(self, idx):
        """
        Creates a {tile_size}x{tile_size} tile centered around the cell instance with the given index.
        Args: 
            idx (int): The index of the cell instance to create a tile for.
        Returns:
            torch.Tensor: The tile centered around the cell instance. Shape: (C, tile_size, tile_size)
            int: The cell instance ID corresponding to the tile.
        """
        cell_id = self.cell_instances[idx]  # Cell instance IDs start from 1
        cell_mask = (self.cell_instance_mask == cell_id).float()

        if cell_mask.sum() == 0:
            raise ValueError(f"Cell instance with ID {cell_id} has no pixels in the cell instance mask.")
        
        # Get the bounding box of the cell instance
        coords = torch.nonzero(cell_mask)
        y_min, x_min = coords.min(dim=0)[0]
        y_max, x_max = coords.max(dim=0)[0]

        # Calculate the center of the bounding box
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2

        # Create a tile centered around the cell instance
        tile_size = self.tile_size // int(self.image_mpp / 0.5)
        half_tile_size = tile_size // 2
        y_start = max(0, y_center - half_tile_size)
        y_end = min(self.tissue_image.shape[1], y_center + half_tile_size)
        x_start = max(0, x_center - half_tile_size)
        x_end = min(self.tissue_image.shape[2], x_center + half_tile_size)

        tile = self.tissue_image[:, y_start:y_end, x_start:x_end]
        tile_cell_mask = cell_mask[y_start:y_end, x_start:x_end]
        tile[:, ~tile_cell_mask.bool()] = self.background_values[:, None]

        # Rescale the image to the desired tile size (effectively, this does the rescaling to 0.5mpp)
        tile = resize(tile, (self.tile_size, self.tile_size))
        return tile, cell_id