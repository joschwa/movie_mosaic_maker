from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AssignmentResult:
    candidate_index_per_cell: list[int]
    passes: int


def _rarity_order(cell_colors: np.ndarray, candidate_colors: np.ndarray) -> np.ndarray:
    """Cell processing order: farthest-from-any-candidate first, so cells with
    rare colors get first pick while the full pool is still available."""
    tree = cKDTree(candidate_colors)
    distances, _ = tree.query(cell_colors, k=1)
    return np.argsort(-distances)


def _build_active_tree(colors: np.ndarray, remaining: np.ndarray) -> tuple[cKDTree | None, np.ndarray]:
    active_indices = np.flatnonzero(remaining > 0)
    if len(active_indices) == 0:
        return None, active_indices
    return cKDTree(colors[active_indices]), active_indices


def assign_candidates(
    cell_colors: np.ndarray,
    candidate_colors: np.ndarray,
    max_reuse: int = 1,
    rebuild_batch_size: int | None = None,
    k_initial: int = 8,
) -> AssignmentResult:
    """Greedily assign each target cell its nearest-color candidate, capping how
    many times any single candidate can be reused.

    cKDTree is immutable, so rebuilding it on every single assignment would be
    too slow for grids of 1000+ cells. Instead candidates are soft-deleted (a
    `remaining` use-budget array) and the tree is only rebuilt in batches --
    every `rebuild_batch_size` exhausted candidates, or immediately if a query
    widened to cover the whole (possibly stale) tree still finds nothing live.

    If the pool is exhausted before every cell is assigned, every candidate's
    budget is reset and assignment continues in a new pass, repeated as many
    times as needed -- `passes` in the result reports how many were needed.
    """
    n_cells = len(cell_colors)
    n_candidates = len(candidate_colors)

    if max_reuse < 1:
        raise ValueError(f"max_reuse must be >= 1, got {max_reuse}")
    if n_candidates == 0:
        raise ValueError("candidate_colors must not be empty")
    if n_cells == 0:
        return AssignmentResult(candidate_index_per_cell=[], passes=0)

    rebuild_batch_size = rebuild_batch_size or max(1, n_candidates // 20)

    remaining = np.full(n_candidates, max_reuse, dtype=int)
    assignment: list[int | None] = [None] * n_cells
    remaining_cells = _rarity_order(cell_colors, candidate_colors).tolist()

    tree, index_map = _build_active_tree(candidate_colors, remaining)
    exhausted_since_rebuild = 0
    passes = 0

    while remaining_cells:
        passes += 1
        still_unassigned = []

        for cell_idx in remaining_cells:
            chosen: int | None = None
            k = min(k_initial, tree.n) if tree is not None else 0

            while chosen is None:
                if tree is None or tree.n == 0:
                    break

                _, tree_indices = tree.query(cell_colors[cell_idx], k=k)
                for t_idx in np.atleast_1d(tree_indices):
                    g_idx = index_map[t_idx]
                    if remaining[g_idx] > 0:
                        chosen = int(g_idx)
                        break

                if chosen is None:
                    if k >= tree.n:
                        tree, index_map = _build_active_tree(candidate_colors, remaining)
                        exhausted_since_rebuild = 0
                        if tree is None or tree.n == 0:
                            break
                        k = min(k_initial, tree.n)
                    else:
                        k = min(k * 4, tree.n)

            if chosen is None:
                still_unassigned.append(cell_idx)
                continue

            assignment[cell_idx] = chosen
            remaining[chosen] -= 1
            if remaining[chosen] == 0:
                exhausted_since_rebuild += 1
                if exhausted_since_rebuild >= rebuild_batch_size:
                    tree, index_map = _build_active_tree(candidate_colors, remaining)
                    exhausted_since_rebuild = 0

        if still_unassigned:
            remaining[:] = max_reuse
            tree, index_map = _build_active_tree(candidate_colors, remaining)
            exhausted_since_rebuild = 0
        remaining_cells = still_unassigned

    return AssignmentResult(candidate_index_per_cell=[int(i) for i in assignment], passes=passes)
