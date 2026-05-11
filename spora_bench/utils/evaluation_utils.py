import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from typing import Dict

def bootstrap_classification_report(
    y_true,
    y_pred,
    n_bootstraps,
    *,
    labels=None,
    target_names=None,
    random_state=None
):
    """
    Bootstrap-based classification report.
    Args:
        y_true (array-like): true labels
        y_pred (array-like): predicted labels
        labels (array-like or None): Optional list of label indices to include and their order. If None, uses sorted unique labels from y_true and y_pred.
        target_names (list or None): Display names corresponding to `labels`.
        n_bootstraps (int): number of bootstraps
        random_state (int or None): random seed for reproducibility
    Returns:
        report (dict): Nested dictionary with precision, recall, f1-score for each class. Leaves of the dictionary are lists of scores across bootstraps.
    """
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # --- label handling (sklearn-like) ---
    if labels is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.asarray(labels)

    n_classes = len(classes)

    if target_names is not None:
        if len(target_names) != n_classes:
            raise ValueError(
                f"target_names must have length {n_classes}, got {len(target_names)}"
            )
        label_to_name = dict(zip(classes, target_names))
    else:
        label_to_name = {c: c for c in classes}

    n = len(y_true)

    # --- storage ---
    report = {
        label_to_name[c]: {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
        for c in classes
    }
    report['macro avg'] = {'precision': [], 'recall': [], 'f1-score': []}
    report['weighted avg'] = {'precision': [], 'recall': [], 'f1-score': []}
    report['accuracy'] = []

    # --- bootstrap loop ---
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping classification report: "):
        idx = rng.integers(0, n, n)
        y_t = y_true[idx]
        y_p = y_pred[idx]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_t,
            y_p,
            labels=classes,
            zero_division=0
        )

        # per-class
        for i, cls in enumerate(classes):
            report[label_to_name[cls]]['precision'].append(precision[i])
            report[label_to_name[cls]]['recall'].append(recall[i])
            report[label_to_name[cls]]['f1-score'].append(f1[i])
            report[label_to_name[cls]]['support'].append(support[i])

        # macro avg
        report['macro avg']['precision'].append(np.mean(precision))
        report['macro avg']['recall'].append(np.mean(recall))
        report['macro avg']['f1-score'].append(np.mean(f1))

        # weighted avg
        total_support = support.sum()
        if total_support > 0:
            weights = support / total_support
        else:
            weights = np.zeros_like(support)

        report['weighted avg']['precision'].append(np.sum(weights * precision))
        report['weighted avg']['recall'].append(np.sum(weights * recall))
        report['weighted avg']['f1-score'].append(np.sum(weights * f1))

        report['accuracy'].append(np.mean(y_t == y_p))

    return report


def fast_bootstrap_classification_report(
    y_true,
    y_pred,
    n_bootstraps,
    *,
    labels=None,
    target_names=None,
    random_state=None
):
    """
    Bootstrap-based classification report.
    Args:
        y_true (array-like): true labels
        y_pred (array-like): predicted labels
        labels (array-like or None): Optional list of label indices to include and their order. If None, uses sorted unique labels from y_true and y_pred.
        target_names (list or None): Display names corresponding to `labels`.
        n_bootstraps (int): number of bootstraps
        random_state (int or None): random seed for reproducibility
    Returns:
        report (dict): Nested dictionary with precision, recall, f1-score for each class. Leaves of the dictionary are lists of scores across bootstraps.
    """
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # --- label handling (sklearn-like) ---
    if labels is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.asarray(labels)

    n_classes = len(classes)

    if target_names is not None:
        if len(target_names) != n_classes:
            raise ValueError(
                f"target_names must have length {n_classes}, got {len(target_names)}"
            )
        label_to_name = dict(zip(classes, target_names))
    else:
        label_to_name = {c: c for c in classes}

    label_to_index = {label: index for index, label in enumerate(classes)}

    n = len(y_true)

    # Encode labels once so each bootstrap can stay in numpy.
    y_true_encoded = np.fromiter((label_to_index[label] for label in y_true), dtype=np.int64, count=n)
    y_pred_encoded = np.fromiter((label_to_index[label] for label in y_pred), dtype=np.int64, count=n)

    # --- storage ---
    report = {
        label_to_name[c]: {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
        for c in classes
    }
    report['macro avg'] = {'precision': [], 'recall': [], 'f1-score': []}
    report['weighted avg'] = {'precision': [], 'recall': [], 'f1-score': []}
    report['accuracy'] = []

    # Cache nested containers used in the hot bootstrap loop.
    class_reports = [report[label_to_name[c]] for c in classes]
    macro_report = report['macro avg']
    weighted_report = report['weighted avg']
    accuracy_report = report['accuracy']

    # --- bootstrap loop ---
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping classification report: "):
        idx = rng.integers(0, n, n)
        y_t = y_true_encoded[idx]
        y_p = y_pred_encoded[idx]

        confusion = np.bincount(
            y_t * n_classes + y_p,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes)

        support = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        true_positive = np.diag(confusion)

        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive, dtype=float),
            where=predicted != 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros_like(true_positive, dtype=float),
            where=support != 0,
        )
        precision_plus_recall = precision + recall
        f1 = np.divide(
            2.0 * precision * recall,
            precision_plus_recall,
            out=np.zeros_like(precision),
            where=precision_plus_recall != 0,
        )

        # per-class
        for i, class_report in enumerate(class_reports):
            class_report['precision'].append(precision[i])
            class_report['recall'].append(recall[i])
            class_report['f1-score'].append(f1[i])
            class_report['support'].append(support[i])

        # macro avg
        macro_report['precision'].append(np.mean(precision))
        macro_report['recall'].append(np.mean(recall))
        macro_report['f1-score'].append(np.mean(f1))

        # weighted avg
        total_support = support.sum()
        if total_support > 0:
            weighted_precision = np.dot(support, precision) / total_support
            weighted_recall = np.dot(support, recall) / total_support
            weighted_f1 = np.dot(support, f1) / total_support
        else:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0

        weighted_report['precision'].append(weighted_precision)
        weighted_report['recall'].append(weighted_recall)
        weighted_report['f1-score'].append(weighted_f1)

        accuracy_report.append(np.mean(y_t == y_p))

    return report

def transform_bootstrap_report_to_df(
        report: Dict,
        n_bootstraps: int,
    ) -> pd.DataFrame:
    """
    Transforms a bootstrap classification report into a pandas DataFrame.
    Args:
        report (Dict): A bootstrap classification report, expected to follow the structure of the output from bootstrap_classification_report.
        n_bootstraps (int): The number of bootstraps.
    Returns:
        pd.DataFrame: The transformed classification report.
    """
    results = []
    for b in range(n_bootstraps):
        for key in report.keys():
            if key not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
                classname = key
                for metric in report[classname].keys():
                    results.append({
                        'metric' : metric,
                        'score' : report[classname][metric][b],
                        'class' : classname,
                        'bootstrap_id' : b,
                    })

        results.append({
            'metric' : 'macro avg f1-score',
            'score' : report['macro avg']['f1-score'][b],
            'class' : 'global',
            'bootstrap_id' : b,
        })
        results.append({
            'metric' : 'weighted avg f1-score',
            'score' : report['weighted avg']['f1-score'][b],
            'class' : 'global',
            'bootstrap_id' : b,
        })
        results.append({
            'metric' : 'accuracy',
            'score' : report['accuracy'][b],
            'class' : 'global',
            'bootstrap_id' : b,
        })
    return pd.DataFrame(results)

def transform_classification_report_to_df(
    report: Dict, 
) -> pd.DataFrame:
    """
    Transforms a classification report into a pandas DataFrame.
    Args:
        report (Dict): A classification report, expected to follow the structure of the output from sklearn's classification_report with output_dict=True.
    Returns:
        pd.DataFrame: The transformed classification report.
    """
    results = []
    for key in report.keys():
        if key not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            classname = key
            for metric in report[classname].keys():
                results.append({
                    'metric' : metric,
                    'score' : report[classname][metric],
                    'class' : classname,
                })
    results.append({
        'metric' : 'macro avg f1-score',
        'score' : report['macro avg']['f1-score'],
        'class' : 'global',
    })
    results.append({
        'metric' : 'weighted avg f1-score',
        'score' : report['weighted avg']['f1-score'],
        'class' : 'global',
    })
    if 'accuracy' in report.keys():
        results.append({
            'metric' : 'accuracy',
            'score' : report['accuracy'],
            'class' : 'global',
        })
    if 'micro avg' in report.keys():
        results.append({
            'metric' : 'micro avg f1-score',
            'score' : report['micro avg']['f1-score'],
            'class' : 'global',
        })
    return pd.DataFrame(results)
    
