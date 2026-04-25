from __future__ import annotations

import functools
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


@functools.lru_cache(maxsize=1)
def latex_rendering_available() -> bool:
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def save_conditional_figure(
    fig: plt.Figure,
    *,
    png_path: Path,
    pdf_path: Path,
    png_dpi: int = 150,
    tight: bool = True,
    close: bool = True,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"} if bool(tight) else {}
    latex_rc = {
        "text.usetex": True,
        "font.family": "serif",
    }
    try:
        if latex_rendering_available():
            with plt.rc_context(latex_rc):
                fig.savefig(png_path, dpi=int(png_dpi), **save_kwargs)
                fig.savefig(pdf_path, **save_kwargs)
        else:
            fig.savefig(png_path, dpi=int(png_dpi), **save_kwargs)
            fig.savefig(pdf_path, **save_kwargs)
    except (RuntimeError, FileNotFoundError):
        fig.savefig(png_path, dpi=int(png_dpi), **save_kwargs)
        fig.savefig(pdf_path, **save_kwargs)
    finally:
        if bool(close):
            plt.close(fig)


def save_conditional_figure_stem(
    fig: plt.Figure,
    *,
    output_stem: Path,
    png_dpi: int = 150,
    tight: bool = True,
    close: bool = True,
) -> dict[str, str]:
    png_path = Path(output_stem).with_suffix(".png")
    pdf_path = Path(output_stem).with_suffix(".pdf")
    save_conditional_figure(
        fig,
        png_path=png_path,
        pdf_path=pdf_path,
        png_dpi=int(png_dpi),
        tight=bool(tight),
        close=bool(close),
    )
    return {"png": str(png_path), "pdf": str(pdf_path)}
