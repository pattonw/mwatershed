"""Accuracy and speed evaluation of the GASP linkage criteria.

Two graph families are evaluated against a known ground truth segmentation at
several noise levels:

* **array graphs** -- every pixel/voxel has affinity edges to a fixed
  neighbourhood (offsets 1, 2 and 4 along each axis, longer offsets noisier),
  segmented with :func:`mwatershed.agglom`.
* **generic graphs** -- nodes scattered in the plane with three edge types that
  may connect the same pair more than once: short range affinities (nearest
  neighbours), long range affinities (noisier) and cosine similarities of
  noisy node embeddings. Segmented with :func:`mwatershed.cluster_edges`.

Noise has two components: i.i.d. Gaussian noise on every edge weight (``sigma``)
and *boundary dropout*, where all boundary edges inside randomly placed patches
are flipped to attractive, which creates correlated merge errors.

Usage::

    uv run --extra dev --with scipy python examples/evaluate_linkages.py \
        --out examples/linkage_eval

Writes ``results.json`` (one record per run) and ``REPORT.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import mwatershed

VARIANTS = [
    ("abs_max", True),
    ("sum", True),
    ("sum", False),
    ("mean", True),
    ("mean", False),
    ("max", True),
    ("max", False),
    ("min", True),
    ("min", False),
]


def variant_name(linkage: str, cannot_link: bool) -> str:
    return linkage if cannot_link else f"{linkage}_nocl"


# ── metrics ──────────────────────────────────────────────────────────


def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.ravel().astype(np.int64)
    b = b.ravel().astype(np.int64)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    return table


def adjusted_rand_index(seg: np.ndarray, gt: np.ndarray) -> float:
    table = contingency(seg, gt)
    n = table.sum()

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_cells = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_cols = comb2(table.sum(axis=0)).sum()
    expected = sum_rows * sum_cols / comb2(n)
    max_index = (sum_rows + sum_cols) / 2.0
    if max_index == expected:
        return 1.0
    return float((sum_cells - expected) / (max_index - expected))


def variation_of_information(seg: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (split, merge): H(seg | gt) and H(gt | seg) in bits."""
    table = contingency(seg, gt).astype(np.float64)
    n = table.sum()
    p = table / n
    p_seg = p.sum(axis=1, keepdims=True)
    p_gt = p.sum(axis=0, keepdims=True)
    nz = p > 0
    h_seg_given_gt = -(
        p[nz] * np.log2(p[nz] / np.broadcast_to(p_gt, p.shape)[nz])
    ).sum()
    h_gt_given_seg = -(
        p[nz] * np.log2(p[nz] / np.broadcast_to(p_seg, p.shape)[nz])
    ).sum()
    return float(h_seg_given_gt), float(h_gt_given_seg)


def score(seg: np.ndarray, gt: np.ndarray) -> dict:
    split, merge = variation_of_information(seg, gt)
    return {
        "ari": adjusted_rand_index(seg, gt),
        "voi_split": split,
        "voi_merge": merge,
        "voi": split + merge,
        "segments": len(np.unique(seg)),
        "gt_segments": len(np.unique(gt)),
    }


# ── ground truth ─────────────────────────────────────────────────────


def voronoi_labels(
    shape: tuple[int, ...], n_cells: int, rng: np.random.Generator
) -> np.ndarray:
    centers = rng.uniform(0, 1, size=(n_cells, len(shape))) * np.array(shape)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    coords = np.stack(grids, axis=-1).reshape(-1, len(shape))
    _, idx = cKDTree(centers).query(coords)
    return (idx + 1).reshape(shape)


def patch_mask(
    coords: np.ndarray, extent: np.ndarray, fraction: float, radius: float, rng
) -> np.ndarray:
    """Boolean mask covering roughly ``fraction`` of the points with random balls."""
    mask = np.zeros(len(coords), dtype=bool)
    if fraction <= 0:
        return mask
    dims = len(extent)
    volume = float(np.prod(extent))
    ball = math.pi * radius**2 if dims == 2 else 4 / 3 * math.pi * radius**3
    n_balls = max(1, int(fraction * volume / ball))
    centers = rng.uniform(0, 1, size=(n_balls, dims)) * extent
    tree = cKDTree(coords)
    for hits in tree.query_ball_point(centers, radius):
        mask[hits] = True
    return mask


# ── array graphs ─────────────────────────────────────────────────────


def array_offsets(dims: int) -> list[tuple[int, ...]]:
    offsets = []
    for r in (1, 2, 4):
        for axis in range(dims):
            off = [0] * dims
            off[axis] = r
            offsets.append(tuple(off))
    return offsets


def affinities_from_labels(
    labels: np.ndarray,
    offsets: list[tuple[int, ...]],
    sigma: float,
    dropout: float,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = labels.shape
    affs = np.zeros((len(offsets),) + shape)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    coords = np.stack(grids, axis=-1).reshape(-1, len(shape))
    radius = 6.0 if len(shape) == 2 else 4.0
    dropout_mask = patch_mask(coords, np.array(shape), dropout, radius, rng).reshape(
        shape
    )
    for i, off in enumerate(offsets):
        sl_u = tuple(slice(max(0, -o), s - max(0, o)) for o, s in zip(off, shape))
        sl_v = tuple(slice(max(0, o), s + min(0, o)) for o, s in zip(off, shape))
        same = labels[sl_u] == labels[sl_v]
        signal = np.where(same, 1.0, -1.0)
        # boundary edges inside dropout patches look attractive
        signal[(~same) & dropout_mask[sl_u]] = 1.0
        # longer range affinities are noisier
        scale = sigma * math.sqrt(max(abs(o) for o in off))
        noise = rng.normal(0.0, scale, size=signal.shape)
        affs[(i,) + sl_u] = signal + noise
    return affs


def run_array(
    dims: int, size: int, n_cells: int, sigma: float, dropout: float, seed: int
) -> list[dict]:
    rng = np.random.default_rng(seed)
    shape = (size,) * dims
    gt = voronoi_labels(shape, n_cells, rng)
    offsets = array_offsets(dims)
    affs = affinities_from_labels(gt, offsets, sigma, dropout, rng)
    records = []
    for linkage, cannot_link in VARIANTS:
        start = time.perf_counter()
        seg = mwatershed.agglom(affs, offsets, linkage=linkage, cannot_link=cannot_link)
        elapsed = time.perf_counter() - start
        rec = {
            "family": f"array{dims}d",
            "shape": list(shape),
            "nodes": int(np.prod(shape)),
            "edges": int(affs.size),
            "sigma": sigma,
            "dropout": dropout,
            "seed": seed,
            "variant": variant_name(linkage, cannot_link),
            "time_s": elapsed,
        }
        rec.update(score(seg, gt))
        records.append(rec)
    return records


# ── generic graphs ───────────────────────────────────────────────────


def make_generic_graph(
    n_nodes: int,
    n_clusters: int,
    sigma: float,
    dropout: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[float, int, int]], np.ndarray, dict]:
    pos = rng.uniform(0, 1, size=(n_nodes, 2))
    centers = rng.uniform(0, 1, size=(n_clusters, 2))
    gt = cKDTree(centers).query(pos)[1] + 1
    tree = cKDTree(pos)
    typical = 1.0 / math.sqrt(n_nodes)  # mean nearest neighbour spacing
    dropout_mask = patch_mask(pos, np.array([1.0, 1.0]), dropout, 6 * typical, rng)

    def signal(u, v):
        same = gt[u] == gt[v]
        s = np.where(same, 1.0, -1.0)
        s[(~same) & dropout_mask[u] & dropout_mask[v]] = 1.0
        return s

    edges: list[tuple[float, int, int]] = []
    counts = {}

    # short range: 6 nearest neighbours
    _, nn = tree.query(pos, k=7)
    u = np.repeat(np.arange(n_nodes), 6)
    v = nn[:, 1:].ravel()
    w = signal(u, v) + rng.normal(0, sigma, size=len(u))
    edges += list(zip(w.tolist(), u.tolist(), v.tolist()))
    counts["short"] = len(u)

    # long range: 3 random nodes within 4x the typical spacing, noisier
    pairs = tree.query_pairs(4 * typical, output_type="ndarray")
    pick = rng.choice(len(pairs), size=min(len(pairs), 3 * n_nodes), replace=False)
    u, v = pairs[pick, 0], pairs[pick, 1]
    w = signal(u, v) + rng.normal(0, 1.5 * sigma, size=len(u))
    edges += list(zip(w.tolist(), u.tolist(), v.tolist()))
    counts["long"] = len(u)

    # embedding similarity: 3 random nodes within 8x the typical spacing
    dim = 16
    cluster_emb = rng.normal(size=(n_clusters + 1, dim))
    emb = (
        cluster_emb[gt] + rng.normal(0, sigma, size=(n_nodes, dim)) * math.sqrt(dim) / 2
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    pairs = tree.query_pairs(8 * typical, output_type="ndarray")
    pick = rng.choice(len(pairs), size=min(len(pairs), 3 * n_nodes), replace=False)
    u, v = pairs[pick, 0], pairs[pick, 1]
    cos = (emb[u] * emb[v]).sum(axis=1)
    w = 2.0 * (cos - 0.5)  # ~ +1 within a cluster, ~ -1 between clusters
    edges += list(zip(w.tolist(), u.tolist(), v.tolist()))
    counts["embedding"] = len(u)

    return edges, gt, counts


def run_generic(
    n_nodes: int, n_clusters: int, sigma: float, dropout: float, seed: int
) -> list[dict]:
    rng = np.random.default_rng(seed)
    edges, gt, counts = make_generic_graph(n_nodes, n_clusters, sigma, dropout, rng)
    records = []
    for linkage, cannot_link in VARIANTS:
        start = time.perf_counter()
        mapping = mwatershed.cluster_edges(
            edges, linkage=linkage, cannot_link=cannot_link
        )
        elapsed = time.perf_counter() - start
        seg = np.arange(n_nodes) + n_nodes  # isolated nodes keep a unique id
        for fragment, segment in mapping:
            seg[fragment] = segment
        rec = {
            "family": "generic",
            "nodes": n_nodes,
            "edges": len(edges),
            "edge_types": counts,
            "sigma": sigma,
            "dropout": dropout,
            "seed": seed,
            "variant": variant_name(linkage, cannot_link),
            "time_s": elapsed,
        }
        rec.update(score(seg, gt))
        records.append(rec)
    return records


# ── report ───────────────────────────────────────────────────────────


def aggregate(records: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(
            (r["family"], r["sigma"], r["dropout"], r["variant"]), []
        ).append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            m: float(np.mean([r[m] for r in rs]))
            for m in ("ari", "voi", "voi_split", "voi_merge", "segments", "time_s")
        }
        out[key]["gt_segments"] = float(np.mean([r["gt_segments"] for r in rs]))
    return out


FAMILY_TITLES = {
    "array2d": "2D array graph",
    "array3d": "3D array graph",
    "generic": "Generic graph",
}


def write_report(records: list[dict], out: Path, config: dict) -> None:
    agg = aggregate(records)
    families = [f for f in FAMILY_TITLES if any(r["family"] == f for r in records)]
    lines = ["# Linkage accuracy and speed evaluation", ""]
    lines.append(
        "Synthetic ground truth segmentations with Gaussian edge noise (`sigma`, applied to a +-1 "
        "signal; longer range edges get proportionally more noise) and boundary dropout "
        "(`dropout`, fraction of the volume covered by patches where boundary edges are flipped "
        "to attractive). Metrics are averaged over seeds. ARI: adjusted Rand index (1 = perfect). "
        "VOI: variation of information in bits (0 = perfect), split + merge. `_nocl` variants "
        "run without cannot-link constraints. Times are wall clock for one call from Python."
    )
    lines.append("")
    lines.append("Configuration: `" + json.dumps(config) + "`")
    lines.append("")
    variants = [variant_name(*v) for v in VARIANTS]
    for family in families:
        example = next(r for r in records if r["family"] == family)
        lines.append(f"## {FAMILY_TITLES[family]}")
        lines.append("")
        desc = f"{example['nodes']} nodes, {example['edges']} edges"
        if "edge_types" in example:
            desc += (
                " ("
                + ", ".join(f"{k}: {v}" for k, v in example["edge_types"].items())
                + ")"
            )
        if "shape" in example:
            desc += f", shape {'x'.join(map(str, example['shape']))}, offsets 1/2/4 per axis"
        lines.append(desc + f", {int(example['gt_segments'])} ground truth segments.")
        lines.append("")
        conditions = sorted(
            {(r["sigma"], r["dropout"]) for r in records if r["family"] == family}
        )

        # best variant summary
        lines.append("### Best variant per condition")
        lines.append("")
        lines.append(
            "| sigma | dropout | best ARI | ARI | abs_max ARI | best VOI | VOI | abs_max VOI |"
        )
        lines.append(
            "|------:|--------:|----------|----:|------------:|----------|----:|------------:|"
        )
        for sigma, dropout in conditions:
            rows = {v: agg[(family, sigma, dropout, v)] for v in variants}
            best_ari = max(rows, key=lambda v: rows[v]["ari"])
            best_voi = min(rows, key=lambda v: rows[v]["voi"])
            lines.append(
                f"| {sigma} | {dropout} | {best_ari} | {rows[best_ari]['ari']:.3f} | {rows['abs_max']['ari']:.3f} "
                f"| {best_voi} | {rows[best_voi]['voi']:.3f} | {rows['abs_max']['voi']:.3f} |"
            )
        lines.append("")

        # full ARI table
        lines.append("### ARI by variant")
        lines.append("")
        lines.append("| sigma | dropout | " + " | ".join(variants) + " |")
        lines.append("|------:|--------:|" + "|".join(["----:"] * len(variants)) + "|")
        for sigma, dropout in conditions:
            cells = [f"{agg[(family, sigma, dropout, v)]['ari']:.3f}" for v in variants]
            lines.append(f"| {sigma} | {dropout} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### VOI (split / merge) by variant")
        lines.append("")
        lines.append("| sigma | dropout | " + " | ".join(variants) + " |")
        lines.append("|------:|--------:|" + "|".join(["----:"] * len(variants)) + "|")
        for sigma, dropout in conditions:
            cells = [
                f"{agg[(family, sigma, dropout, v)]['voi_split']:.2f} / {agg[(family, sigma, dropout, v)]['voi_merge']:.2f}"
                for v in variants
            ]
            lines.append(f"| {sigma} | {dropout} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### Segments by variant (ground truth in parentheses)")
        lines.append("")
        lines.append("| sigma | dropout | " + " | ".join(variants) + " |")
        lines.append("|------:|--------:|" + "|".join(["----:"] * len(variants)) + "|")
        for sigma, dropout in conditions:
            cells = [
                f"{agg[(family, sigma, dropout, v)]['segments']:.0f}" for v in variants
            ]
            gt_n = agg[(family, sigma, dropout, variants[0])]["gt_segments"]
            lines.append(
                f"| {sigma} | {dropout} ({gt_n:.0f}) | " + " | ".join(cells) + " |"
            )
        lines.append("")

        lines.append("### Time in ms by variant")
        lines.append("")
        lines.append("| sigma | dropout | " + " | ".join(variants) + " |")
        lines.append("|------:|--------:|" + "|".join(["----:"] * len(variants)) + "|")
        for sigma, dropout in conditions:
            cells = [
                f"{agg[(family, sigma, dropout, v)]['time_s'] * 1e3:.0f}"
                for v in variants
            ]
            lines.append(f"| {sigma} | {dropout} | " + " | ".join(cells) + " |")
        lines.append("")

    (out / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("examples/linkage_eval"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.6, 0.9, 1.2])
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    parser.add_argument("--size-2d", type=int, default=256)
    parser.add_argument("--cells-2d", type=int, default=120)
    parser.add_argument("--size-3d", type=int, default=48)
    parser.add_argument("--cells-3d", type=int, default=100)
    parser.add_argument("--generic-nodes", type=int, default=30000)
    parser.add_argument("--generic-clusters", type=int, default=150)
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["array2d", "array3d", "generic"]
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    records: list[dict] = []
    total_start = time.perf_counter()
    for sigma in args.sigmas:
        for dropout in args.dropouts:
            for seed in range(args.seeds):
                if "array2d" not in args.skip:
                    records += run_array(
                        2, args.size_2d, args.cells_2d, sigma, dropout, seed
                    )
                if "array3d" not in args.skip:
                    records += run_array(
                        3, args.size_3d, args.cells_3d, sigma, dropout, seed
                    )
                if "generic" not in args.skip:
                    records += run_generic(
                        args.generic_nodes, args.generic_clusters, sigma, dropout, seed
                    )
            print(
                f"sigma={sigma} dropout={dropout} done "
                f"({time.perf_counter() - total_start:.0f}s elapsed)",
                flush=True,
            )
            (args.out / "results.json").write_text(json.dumps(records, indent=1))
    write_report(records, args.out, config)
    print(f"wrote {args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
