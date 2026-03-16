from __future__ import annotations

from pathlib import Path

import pandas as pd

from clin_omics.visualization import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_regression_residuals,
    plot_roc_curve,
)


def test_plot_roc_curve_saves_png_and_svg(tmp_path: Path) -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.4, 0.6, 0.9])
    out_prefix = tmp_path / 'roc'

    fig, ax = plot_roc_curve(y_true=y_true, y_score=y_score, out_prefix=out_prefix)

    assert fig is not None
    assert ax is not None
    assert (tmp_path / 'roc.png').exists()
    assert (tmp_path / 'roc.svg').exists()


def test_plot_pr_curve_saves_png_and_svg(tmp_path: Path) -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.4, 0.6, 0.9])
    out_prefix = tmp_path / 'pr'

    fig, ax = plot_pr_curve(y_true=y_true, y_score=y_score, out_prefix=out_prefix)

    assert fig is not None
    assert ax is not None
    assert (tmp_path / 'pr.png').exists()
    assert (tmp_path / 'pr.svg').exists()


def test_plot_confusion_matrix_saves_png_and_svg(tmp_path: Path) -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 1, 1])
    out_prefix = tmp_path / 'cm'

    fig, ax = plot_confusion_matrix(y_true=y_true, y_pred=y_pred, out_prefix=out_prefix)

    assert fig is not None
    assert ax is not None
    assert (tmp_path / 'cm.png').exists()
    assert (tmp_path / 'cm.svg').exists()


def test_plot_regression_residuals_saves_png_and_svg(tmp_path: Path) -> None:
    y_true = pd.Series([0.0, 1.0, 2.0, 3.0])
    y_pred = pd.Series([0.2, 0.9, 2.1, 2.8])
    out_prefix = tmp_path / 'residuals'

    fig, ax = plot_regression_residuals(y_true=y_true, y_pred=y_pred, out_prefix=out_prefix)

    assert fig is not None
    assert ax is not None
    assert (tmp_path / 'residuals.png').exists()
    assert (tmp_path / 'residuals.svg').exists()
