from typing import Tuple

import numpy as np


def align_frames_procrustes(U_stack: np.ndarray, U_ref: np.ndarray) -> np.ndarray:
    aligned = []
    for i in range(U_stack.shape[0]):
        Ui = U_stack[i]
        M = Ui.T @ U_ref
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        aligned.append(Ui @ R)
    return np.stack(aligned, axis=0)


def align_frames_procrustes_with_rotations(
    U_stack: np.ndarray, U_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    aligned = []
    rotations = []
    for i in range(U_stack.shape[0]):
        Ui = U_stack[i]
        M = Ui.T @ U_ref
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        aligned.append(Ui @ R)
        rotations.append(R)
    return np.stack(aligned, axis=0), np.stack(rotations, axis=0)


def apply_rotations_to_embeddings(embeddings: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings)
    rot = np.asarray(rotations)
    if emb.shape[0] != rot.shape[0]:
        raise ValueError('embeddings and rotations must share the first dimension (time).')
    return np.einsum('tnk,tkj->tnj', emb, rot)
