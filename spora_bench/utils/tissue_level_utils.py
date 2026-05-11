import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class PatientBagDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, patient_to_tokens, patient_to_label):
        self.patient_ids = patient_ids
        self.patient_to_tokens = patient_to_tokens
        self.patient_to_label = patient_to_label

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        features = self.patient_to_tokens[patient_id]
        label = self.patient_to_label[patient_id]
        return features, label


def collate_patient_bags(batch):
    features_list, labels = zip(*batch)

    lengths = torch.as_tensor([x.size(0) for x in features_list], dtype=torch.long)
    padded_features = pad_sequence(features_list, batch_first=True)  # pads with 0.0

    max_len = padded_features.size(1)
    attn_mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    labels = torch.as_tensor(labels, dtype=torch.long)
    return padded_features, attn_mask, labels


def get_targets(ds_config: OmegaConf):
    metadata = pd.read_parquet(ds_config.path + "/metadata/tissues.parquet")
    labels = {}
    for downstream_name, downstream_task in ds_config.benchmarks.tissue_level.items():
        column = downstream_task.label_col
        filter_values = downstream_task.get("excluded_classes", [])
        target_data = metadata[column].astype(str)
        target_data = target_data.apply(lambda x: x if x not in filter_values and x != 'None' else None)
        labels_task = dict(zip(metadata["tissue_id"], target_data))
        labels_task = {tid: str(label) for tid, label in labels_task.items() if label is not None}
        labels[downstream_name] = labels_task
    return labels

def get_patient_splits(ds_config: OmegaConf):
    metadata = pd.read_parquet(ds_config.path + "/metadata/tissues.parquet")
    metadata = metadata[["patient_id", "split"]].set_index("patient_id")
    return metadata.to_dict()["split"]

def config_has_tissue_level_benchmark(config):
    if "benchmarks" not in config:
        return False
    if "tissue_level" not in config.benchmarks:
        return False
    return True

def abmil_train_test(config: OmegaConf, 
                     patient_to_tokens: dict, 
                     patient_to_label: dict, 
                     train_ids: list, 
                     val_ids: list, 
                     test_tids: list
                     ) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """ Train and test the ABMIL model on the given train, val, and test splits. Returns the trained model, test predictions, and test labels.
    Args:
        config: The configuration object containing training parameters.
        patient_to_tokens: A dictionary mapping patient IDs to their corresponding tile token tensors.
        patient_to_label: A dictionary mapping patient IDs to their corresponding labels.
        train_ids: A list of patient IDs in the training set.
        val_ids: A list of patient IDs in the validation set.
        test_tids: A list of tissue IDs in the test set (tissue IDs are in the format "tissue_{patient_id}").
    """

    train_dataset = PatientBagDataset(train_ids, patient_to_tokens, patient_to_label)
    val_dataset = PatientBagDataset(val_ids, patient_to_tokens, patient_to_label)
    test_dataset = PatientBagDataset(test_tids, patient_to_tokens, patient_to_label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_patient_bags,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_patient_bags,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_patient_bags,
        persistent_workers=True,
    )

    sample_patient = train_ids[0]
    feature_dim = patient_to_tokens[sample_patient].shape[1]
    n_classes = len(set(patient_to_label.values()))

    # Instantiate the ABMIL model
    model = instantiate(config.downstream_model, input_dim=feature_dim, num_classes=n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    model.train_model(
        train_loader,
        val_loader,
        num_epochs=config.training.max_epochs,
        optimizer=optimizer,
        loss_fn=criterion,
    )

    preds, targets, _ = model.compute_predictions_and_truth(test_loader, criterion)
    return model, np.array(preds), np.array(targets)