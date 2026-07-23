from collections import Counter

import numpy as np
import pytest
from scipy.spatial import cKDTree

from movie_mosaic_maker.assignment import _rarity_order, assign_candidates


def _slow_reference_assign(cell_colors: np.ndarray, candidate_colors: np.ndarray, max_reuse: int) -> tuple[list[int], int]:
    """Obviously-correct O(n^2) reference: always scans every live candidate
    fresh (no batching/staleness tricks) to find the true nearest neighbor."""
    n_candidates = len(candidate_colors)
    remaining = [max_reuse] * n_candidates
    order = list(_rarity_order(cell_colors, candidate_colors))
    assignment: list[int | None] = [None] * len(cell_colors)
    remaining_cells = order
    passes = 0

    while remaining_cells:
        passes += 1
        still_unassigned = []
        for cell_idx in remaining_cells:
            best_candidate = None
            best_dist = None
            for ci in range(n_candidates):
                if remaining[ci] <= 0:
                    continue
                dist = float(np.linalg.norm(candidate_colors[ci] - cell_colors[cell_idx]))
                if best_dist is None or dist < best_dist:
                    best_candidate, best_dist = ci, dist
            if best_candidate is None:
                still_unassigned.append(cell_idx)
                continue
            assignment[cell_idx] = best_candidate
            remaining[best_candidate] -= 1
        if still_unassigned:
            remaining = [max_reuse] * n_candidates
        remaining_cells = still_unassigned

    return [i for i in assignment], passes  # type: ignore[misc]


def _total_cost(cell_colors: np.ndarray, candidate_colors: np.ndarray, assignment: list[int]) -> float:
    return float(sum(np.linalg.norm(candidate_colors[a] - cell_colors[i]) for i, a in enumerate(assignment)))


def test_rarity_order_processes_farthest_color_first() -> None:
    candidates = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    cells = np.array(
        [
            [0.0, 0.0, 0.0],  # exact match, distance 0
            [100.0, 0.0, 0.0],  # far from everything
            [5.0, 0.0, 0.0],  # equidistant-ish, moderate distance
        ]
    )
    order = _rarity_order(cells, candidates)
    assert order[0] == 1  # the far-away cell must be processed first


def test_assign_candidates_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        assign_candidates(np.zeros((2, 3)), np.zeros((5, 3)), max_reuse=0)
    with pytest.raises(ValueError):
        assign_candidates(np.zeros((2, 3)), np.zeros((0, 3)), max_reuse=1)


def test_assign_candidates_empty_cells_returns_empty_result() -> None:
    result = assign_candidates(np.zeros((0, 3)), np.zeros((5, 3)), max_reuse=1)
    assert result.candidate_index_per_cell == []
    assert result.passes == 0


def test_assign_candidates_every_cell_gets_assigned() -> None:
    rng = np.random.default_rng(0)
    cells = rng.uniform(0, 100, size=(40, 3))
    candidates = rng.uniform(0, 100, size=(15, 3))
    result = assign_candidates(cells, candidates, max_reuse=3)
    assert len(result.candidate_index_per_cell) == 40
    assert all(a is not None for a in result.candidate_index_per_cell)


def test_assign_candidates_respects_max_reuse_within_a_single_pass() -> None:
    rng = np.random.default_rng(1)
    candidates = rng.uniform(0, 100, size=(10, 3))
    cells = rng.uniform(0, 100, size=(10, 3))  # exactly pool_size * max_reuse
    result = assign_candidates(cells, candidates, max_reuse=1)
    assert result.passes == 1
    counts = Counter(result.candidate_index_per_cell)
    assert all(count <= 1 for count in counts.values())


def test_assign_candidates_multi_pass_matches_expected_pass_count() -> None:
    rng = np.random.default_rng(2)
    candidates = rng.uniform(0, 100, size=(3, 3))
    cells = rng.uniform(0, 100, size=(7, 3))  # ceil(7/3) = 3 passes needed
    result = assign_candidates(cells, candidates, max_reuse=1)
    assert result.passes == 3
    assert len(result.candidate_index_per_cell) == 7


def test_assign_candidates_total_reuse_bounded_by_max_reuse_times_passes() -> None:
    rng = np.random.default_rng(3)
    candidates = rng.uniform(0, 100, size=(4, 3))
    cells = rng.uniform(0, 100, size=(15, 3))
    max_reuse = 2
    result = assign_candidates(cells, candidates, max_reuse=max_reuse)
    counts = Counter(result.candidate_index_per_cell)
    assert all(count <= max_reuse * result.passes for count in counts.values())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_assign_candidates_batched_matches_slow_reference_quality(seed: int) -> None:
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(0, 100, size=(12, 3))
    cells = rng.uniform(0, 100, size=(30, 3))
    max_reuse = 2

    fast_result = assign_candidates(cells, candidates, max_reuse=max_reuse, rebuild_batch_size=3, k_initial=2)
    slow_assignment, slow_passes = _slow_reference_assign(cells, candidates, max_reuse=max_reuse)

    fast_cost = _total_cost(cells, candidates, fast_result.candidate_index_per_cell)
    slow_cost = _total_cost(cells, candidates, slow_assignment)

    assert fast_result.passes == slow_passes
    assert fast_cost == pytest.approx(slow_cost, rel=1e-6)


def test_assign_candidates_single_candidate_many_cells() -> None:
    candidates = np.array([[50.0, 0.0, 0.0]])
    cells = np.array([[0.0, 0.0, 0.0]] * 5)
    result = assign_candidates(cells, candidates, max_reuse=2)
    assert len(result.candidate_index_per_cell) == 5
    assert all(a == 0 for a in result.candidate_index_per_cell)
    assert result.passes == 3  # ceil(5/2)
