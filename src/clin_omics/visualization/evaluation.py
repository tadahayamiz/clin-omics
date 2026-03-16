from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from clin_omics.visualization.save import save_figure
from clin_omics.visualization.style import PlotConfig, resolve_plot_config


def plot_roc_curve(
    *,
    y_true: pd.Series | np.ndarray | list,
    y_score: pd.Series | np.ndarray | list,
    title: str | None = None,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    resolved = resolve_plot_config(config)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=resolved.label_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=resolved.label_fontsize)
    ax.set_title(title or 'ROC Curve', fontsize=resolved.title_fontsize)
    ax.tick_params(axis='both', labelsize=resolved.tick_fontsize)
    ax.legend(fontsize=resolved.legend_fontsize)
    save_figure(fig, out_prefix, config=resolved)
    return fig, ax


def plot_pr_curve(
    *,
    y_true: pd.Series | np.ndarray | list,
    y_score: pd.Series | np.ndarray | list,
    title: str | None = None,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    resolved = resolve_plot_config(config)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel('Recall', fontsize=resolved.label_fontsize)
    ax.set_ylabel('Precision', fontsize=resolved.label_fontsize)
    ax.set_title(title or 'Precision-Recall Curve', fontsize=resolved.title_fontsize)
    ax.tick_params(axis='both', labelsize=resolved.tick_fontsize)
    ax.legend(fontsize=resolved.legend_fontsize)
    save_figure(fig, out_prefix, config=resolved)
    return fig, ax


def plot_confusion_matrix(
    *,
    y_true: pd.Series | np.ndarray | list,
    y_pred: pd.Series | np.ndarray | list,
    title: str | None = None,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    resolved = resolve_plot_config(config)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=resolved.label_fontsize)
    ax.set_ylabel('True', fontsize=resolved.label_fontsize)
    ax.set_title(title or 'Confusion Matrix', fontsize=resolved.title_fontsize)
    ax.tick_params(axis='both', labelsize=resolved.tick_fontsize)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=resolved.tick_fontsize)

    save_figure(fig, out_prefix, config=resolved)
    return fig, ax


def plot_regression_residuals(
    *,
    y_true: pd.Series | np.ndarray | list,
    y_pred: pd.Series | np.ndarray | list,
    title: str | None = None,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    resolved = resolve_plot_config(config)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr

    fig, ax = plt.subplots(figsize=resolved.figsize)
    ax.scatter(y_pred_arr, residuals, s=40)
    ax.axhline(0.0, linestyle='--')
    ax.set_xlabel('Predicted', fontsize=resolved.label_fontsize)
    ax.set_ylabel('Residual (true - pred)', fontsize=resolved.label_fontsize)
    ax.set_title(title or 'Regression Residuals', fontsize=resolved.title_fontsize)
    ax.tick_params(axis='both', labelsize=resolved.tick_fontsize)
    save_figure(fig, out_prefix, config=resolved)
    return fig, ax
