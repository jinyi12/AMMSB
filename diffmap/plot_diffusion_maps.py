"""Module for plotting diffusion maps and associated diagnostics.

"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def get_rows_and_columns(num_plots: int) -> Tuple[int, int]:
    """Get optimal number of rows and columns to display figures.

    Parameters
    ----------
    num_plots : int
        Number of subplots

    Returns
    -------
    rows : int
        Optimal number of rows.
    cols : int
        Optimal number of columns.

    """
    if num_plots <= 10:
        layouts = {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
            6: (2, 3),
            7: (2, 4),
            8: (2, 4),
            9: (3, 9),
            10: (2, 5),
        }
        rows, cols = layouts[num_plots]
    else:
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = rows

    return rows, cols


def plot_diffusion_maps(
    data: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> None:
    """Plot results.

    Plots three figures. The first one is shows the modulus of the
    spectrum of the kernel in the diffusion map calculation.  The
    second displays the original (2D) data colored by the value of
    each diffusion map.  The third figure displays the data, as
    trasnformed by the first two diffusion maps.

    Parameters
    ----------
    data : np.ndarray
        Original (or downsampled) data set.
    eigenvalues : array
        Eigenvalues of the kernel matrix.
    eigenvectors : array
        Eigenvectors of the kernel matrix. The second axis indexes
        each vector.

    """
    x, y = data[:, 0], data[:, 1]

    num_eigenvectors = max(eigenvectors.shape[1], 10)

    plt.figure(1)
    plt.clf()
    plt.step(np.arange(1, eigenvalues.shape[0]), np.abs(eigenvalues[1:]))
    plt.xticks(range(1, eigenvalues.shape[0]))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('| Eigenvalue |')
    plt.title('Eigenvalues')

    plt.figure(2)
    plt.clf()
    rows, cols = get_rows_and_columns(num_eigenvectors)
    for k in range(1, eigenvectors.shape[1] + 1):
        plt.subplot(rows, cols, k)
        plt.scatter(
            x, y, c=eigenvectors[:, k - 1], cmap='RdBu_r', rasterized=True
        )
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.axis('off')
        plt.title(f'$\\psi_{{{k}}}$')

    plt.figure(3)
    plt.clf()
    plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c='black', alpha=0.5)
    plt.xlabel('$\\psi_1$')
    plt.ylabel('$\\psi_2$')
    plt.title('Data set in diffusion map space')

    # plt.tight_layout()
    plt.show()