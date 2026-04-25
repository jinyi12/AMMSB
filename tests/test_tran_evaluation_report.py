from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation import report


def test_conditioned_qualitative_panel_supports_rowwise_colorbars_and_difference_row(tmp_path):
    qualitative_examples = {
        "generated_fields": np.asarray(
            [
                [50.0, 52.0, 54.0, 56.0],
                [51.0, 53.0, 55.0, 57.0],
                [52.0, 54.0, 56.0, 58.0],
            ],
            dtype=np.float32,
        ),
        "coarsened_fields": np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        ),
        "condition_fields": np.asarray(
            [
                [-1.0, 0.0, 4.0, 5.0],
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 2.0, 6.0, 7.0],
            ],
            dtype=np.float32,
        ),
    }

    captured: dict[str, object] = {}

    def _capture(fig, out_dir, name, *, png_dpi=150):
        del out_dir, png_dpi
        captured["fig"] = fig
        captured["name"] = name

    original_save = report._save_fig
    report._save_fig = _capture
    try:
        report._plot_conditioned_qualitative_panel(
            qualitative_examples,
            resolution=2,
            out_dir=tmp_path,
            name="qualitative_panel",
            include_difference_row=True,
            rowwise_colorbars=True,
        )
    finally:
        report._save_fig = original_save

    fig = captured["fig"]
    assert captured["name"] == "qualitative_panel"

    main_axes = [ax for ax in fig.axes if ax.images]
    assert len(main_axes) == 12
    assert len(fig.axes) == 16

    main_axes = np.asarray(main_axes, dtype=object).reshape(4, 3)
    assert main_axes[0, 0].get_ylabel() == "Generated"
    assert main_axes[1, 0].get_ylabel() == "Coarsened"
    assert main_axes[2, 0].get_ylabel() == "GT / Cond."
    assert main_axes[3, 0].get_ylabel() == "Difference"

    cmaps = [ax.images[0].get_cmap().name for ax in main_axes.ravel()]
    assert cmaps.count(report.CMAP_FIELD) == 9
    assert cmaps.count(report.CMAP_DIFF) == 3

    generated_clim = main_axes[0, 0].images[0].get_clim()
    coarsened_clim = main_axes[1, 0].images[0].get_clim()
    condition_clim = main_axes[2, 0].images[0].get_clim()
    diff_clim = main_axes[3, 0].images[0].get_clim()

    assert generated_clim != coarsened_clim
    assert coarsened_clim != condition_clim
    assert diff_clim[0] == pytest.approx(-2.0)
    assert diff_clim[1] == pytest.approx(2.0)


def test_plot_coarse_consistency_global_qualitative_uses_publication_row_labels(tmp_path):
    coarse_qualitative_results = {
        "conditioned_global_return": {
            "generated_fields": np.asarray(
                [
                    [50.0, 52.0, 54.0, 56.0],
                    [51.0, 53.0, 55.0, 57.0],
                    [52.0, 54.0, 56.0, 58.0],
                ],
                dtype=np.float32,
            ),
            "coarsened_fields": np.asarray(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                ],
                dtype=np.float32,
            ),
            "condition_fields": np.asarray(
                [
                    [-1.0, 0.0, 4.0, 5.0],
                    [0.0, 1.0, 5.0, 6.0],
                    [1.0, 2.0, 6.0, 7.0],
                ],
                dtype=np.float32,
            ),
            "transfer_metadata": {
                "source_H": 1.0,
                "target_H": 6.0,
                "ridge_lambda": 1e-8,
            },
        }
    }

    captured: dict[str, object] = {}

    def _capture(fig, out_dir, name, *, png_dpi=150):
        del out_dir, png_dpi
        captured["fig"] = fig
        captured["name"] = name

    original_save = report._save_fig
    report._save_fig = _capture
    try:
        report.plot_coarse_consistency_global_qualitative(
            coarse_qualitative_results,
            resolution=2,
            out_dir=tmp_path,
        )
    finally:
        report._save_fig = original_save

    fig = captured["fig"]
    assert captured["name"] == "fig1c_coarse_consistency_global_qualitative"

    labels = [text.get_text() for text in fig.texts]
    assert labels == [
        r"$\widetilde U_{H=1}$",
        r"$\mathcal{T}^{\mathrm{reg}}_{6\leftarrow1}\widetilde U_{H=1}$",
        r"$U_{H=6}$",
        r"$\mathcal{T}^{\mathrm{reg}}_{6\leftarrow1}\widetilde U_{H=1} - U_{H=6}$",
    ]

    width, height = fig.get_size_inches()
    assert width == pytest.approx(5.6)
    assert height == pytest.approx(5.15)


def test_plot_coarse_consistency_interval_qualitative_uses_revised_row_labels(tmp_path):
    coarse_results = {
        "conditioned_interval": {
            "pair_H6_to_H4": {
                "pair_metadata": {
                    "display_label": "H=6 -> H=4",
                }
            }
        }
    }
    coarse_qualitative_results = {
        "conditioned_interval": {
            "pair_H6_to_H4": {
                "generated_fields": np.asarray(
                    [
                        [50.0, 52.0, 54.0, 56.0],
                        [51.0, 53.0, 55.0, 57.0],
                    ],
                    dtype=np.float32,
                ),
                "coarsened_fields": np.asarray(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
                "condition_fields": np.asarray(
                    [
                        [-1.0, 0.0, 4.0, 5.0],
                        [0.0, 1.0, 5.0, 6.0],
                    ],
                    dtype=np.float32,
                ),
                "transfer_metadata": {
                    "source_H": 4.0,
                    "target_H": 6.0,
                    "ridge_lambda": 1e-8,
                },
            }
        }
    }

    captured: dict[str, object] = {}

    def _capture(fig, out_dir, name, *, png_dpi=150):
        del out_dir, png_dpi
        captured["fig"] = fig
        captured["name"] = name

    original_save = report._save_fig
    report._save_fig = _capture
    try:
        report.plot_coarse_consistency_interval_qualitative(
            coarse_results,
            coarse_qualitative_results,
            resolution=2,
            out_dir=tmp_path,
        )
    finally:
        report._save_fig = original_save

    fig = captured["fig"]
    assert captured["name"] == "fig1d_coarse_consistency_pair_H6_to_H4_qualitative"

    main_axes = [ax for ax in fig.axes if ax.images]
    main_axes = np.asarray(main_axes, dtype=object).reshape(4, 2)
    assert main_axes[0, 0].get_ylabel() == r"$\widetilde U_{H=4}$"
    assert main_axes[1, 0].get_ylabel() == r"$\mathcal{T}^{\mathrm{reg}}_{6\leftarrow4}\widetilde U_{H=4}$"
    assert main_axes[2, 0].get_ylabel() == r"$U_{H=6}$"
    assert main_axes[3, 0].get_ylabel() == r"$\mathcal{T}^{\mathrm{reg}}_{6\leftarrow4}\widetilde U_{H=4} - U_{H=6}$"
