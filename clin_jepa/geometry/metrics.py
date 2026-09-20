"""Cohort-separation statistics of one encoder, computed on its per-hour state embeddings."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

CONFIGURATIONS = [
    ("main", "all", "l2"), ("main", "all", "cos"), ("main", "all", "pca"), ("main", "subsample", "l2"),
    ("delta2", "all", "l2"), ("delta4", "all", "l2"), ("stable_std_only", "all", "l2"),
]


def load_trajectories(cache_dir: Path, stay_ids: list[int], hours: int, pool_cfg: dict) -> tuple:
    """Admission trajectories ``(n, hours, D)`` of the given stays and a random pool of test-split states."""
    position = {stay_id: i for i, stay_id in enumerate(stay_ids)}
    rng = np.random.RandomState(pool_cfg["seed"])
    trajectories, pool, found = None, [], set()
    for path in sorted(cache_dir.glob("embeddings_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        z = shard["z_state_embeddings"]
        if trajectories is None:
            trajectories = np.zeros((len(stay_ids), hours, z.shape[1]), dtype=np.float32)
        pool.append(z[rng.choice(len(z), min(pool_cfg["rows_per_shard"], len(z)), replace=False)].float().numpy())
        for w in shard["windows"]:
            i = position.get(int(w["stay_id"]))
            if i is not None and int(w["start_step"]) == 0 and int(w["length"]) >= hours:
                trajectories[i] = z[w["emb_indices_state"][:hours].long()].float().numpy()
                found.add(i)
    if len(found) != len(stay_ids):
        raise ValueError(f"{len(stay_ids) - len(found)} cohort stays have no admission window in {cache_dir}")
    return trajectories, np.concatenate(pool)


def unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


class Space:
    """Euclidean (``l2``), cosine (``cos``) or Euclidean after a PCA projection (``pca``)."""

    def __init__(self, name: str, pca: PCA | None = None) -> None:
        self.name, self.pca = name, pca

    def transform(self, a: np.ndarray) -> np.ndarray:
        if self.name == "cos":
            return unit(a)
        if self.name == "pca":
            flat = self.pca.transform(a.reshape(-1, a.shape[-1])).astype(np.float32)
            return flat.reshape(*a.shape[:-1], flat.shape[-1])
        return a

    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.name == "cos":
            return 1.0 - (unit(a) * unit(b)).sum(-1)
        return np.linalg.norm(a - b, axis=-1)

    def pairwise(self, a: np.ndarray) -> np.ndarray:
        a = torch.from_numpy(np.ascontiguousarray(a))
        if self.name == "cos":
            u = torch.nn.functional.normalize(a, dim=1)
            return (1.0 - u @ u.T).numpy()
        squared = (a * a).sum(1)
        return torch.sqrt(torch.clamp(squared[:, None] + squared[None, :] - 2.0 * (a @ a.T), min=0.0)).numpy()


def typical_distance(space: Space, pool: np.ndarray, cfg: dict) -> float:
    """Mean distance between two randomly drawn test-split states."""
    rng = np.random.RandomState(cfg["seed"])
    i, j = rng.choice(len(pool), cfg["pairs"], replace=True), rng.choice(len(pool), cfg["pairs"], replace=True)
    return float(space.distance(pool[i[i != j]], pool[j[i != j]]).mean())


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    return float((x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))


def balanced_accuracy(correct: np.ndarray, y: np.ndarray) -> float:
    return float((correct[y == 1].mean() + correct[y == 0].mean()) / 2)


def leave_one_out_means(v: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per stay, the mean of each cohort without that stay."""
    means = []
    for value in (1, 0):
        member = y == value
        total = np.repeat(v[member].sum(0)[None], len(v), 0)
        total[member] -= v[member]
        means.append((total / np.where(member, member.sum() - 1, member.sum())[:, None]).astype(np.float32))
    return means[0], means[1]


def resample(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    det, sta = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return det[rng.integers(0, len(det), len(det))], sta[rng.integers(0, len(sta), len(sta))]


def curve_interval(per_stay: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int, balanced: bool) -> list:
    """95 % interval of a per-hour mean of ``per_stay (hours, n)`` under resampling of stays within cohorts."""
    values = np.zeros((draws, len(per_stay)))
    for b in range(draws):
        det, sta = resample(y, rng)
        if balanced:
            values[b] = (per_stay[:, det].mean(1) + per_stay[:, sta].mean(1)) / 2
        else:
            values[b] = per_stay[:, np.concatenate([det, sta])].mean(1)
    return np.percentile(values, [2.5, 97.5], axis=0).tolist()


def multiplicity(rng: np.random.Generator, n: int, draws: int) -> np.ndarray:
    weights = np.zeros((draws, n), dtype=np.float32)
    for b in range(draws):
        weights[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return weights / n


def direction(a: np.ndarray, y: np.ndarray, hour: int) -> tuple[np.ndarray, float]:
    """Whether each stay's displacement since admission is more parallel to its own cohort's mean; cohort cosine."""
    v = a[:, hour] - a[:, 0]
    mean_det, mean_sta = leave_one_out_means(v, y)
    u = unit(v)
    predicted = ((u * unit(mean_det)).sum(1) > (u * unit(mean_sta)).sum(1)).astype(int)
    return (predicted == y).astype(np.float32), float(unit(v[y == 1].mean(0)) @ unit(v[y == 0].mean(0)))


def analyse(space: Space, a: np.ndarray, y: np.ndarray, scale: float, neighbours: list[int],
            rng: np.random.Generator | None = None, draws: int = 0) -> dict:
    hours = a.shape[1]
    det, sta = a[y == 1], a[y == 0]
    out = {"n_deteriorating": len(det), "n_stable": len(sta)}

    net = space.distance(a[:, -1], a[:, 0])
    net_det, net_sta = net[y == 1], net[y == 0]
    out["net_displacement"] = {
        "deteriorating_mean": float(net_det.mean() / scale), "stable_mean": float(net_sta.mean() / scale),
        "pooled_sd": float(np.sqrt((net_det.var(ddof=1) + net_sta.var(ddof=1)) / 2) / scale),
        "cohens_d": cohens_d(net_det, net_sta),
        "p_value": float(mannwhitneyu(net_det, net_sta, alternative="greater")[1]),
    }
    from_start = space.distance(a, a[:, :1]) / scale
    out["from_start"] = {"deteriorating": from_start[y == 1].mean(0).tolist(),
                         "stable": from_start[y == 0].mean(0).tolist()}
    out["divergence"] = (space.distance(det.mean(0), sta.mean(0)) / scale).tolist()

    k_max = min(max(neighbours), len(a) - 1)
    same_cohort = np.zeros((hours, len(a), k_max), dtype=bool)
    centroid_correct = np.zeros((hours, len(a)), dtype=np.float32)
    for t in range(hours):
        d = space.pairwise(a[:, t])
        np.fill_diagonal(d, np.inf)
        nearest = np.argpartition(d, k_max, axis=1)[:, :k_max]
        nearest = np.take_along_axis(nearest, np.argsort(np.take_along_axis(d, nearest, 1), axis=1), 1)
        same_cohort[t] = y[nearest] == y[:, None]
        centroid_det, centroid_sta = leave_one_out_means(a[:, t], y)
        nearer_det = space.distance(a[:, t], centroid_det) < space.distance(a[:, t], centroid_sta)
        centroid_correct[t] = nearer_det.astype(int) == y
    n, n_det, n_sta = len(y), len(det), len(sta)
    out["knn_chance"] = float(n_det / n * (n_det - 1) / (n - 1) + n_sta / n * (n_sta - 1) / (n - 1))
    purity = {k: same_cohort[:, :, :k].mean(2) for k in neighbours}
    out["knn_purity"] = {str(k): purity[k].mean(1).tolist() for k in neighbours}
    out["centroid_accuracy"] = [balanced_accuracy(centroid_correct[t], y) for t in range(hours)]

    direction_correct, cosine = direction(a, y, hours - 1)
    out["direction_accuracy_last"] = balanced_accuracy(direction_correct, y)
    out["direction_cosine_last"] = cosine

    if draws:
        out["net_displacement"]["per_stay"] = {"deteriorating": (net_det / scale).tolist(),
                                               "stable": (net_sta / scale).tolist()}
        boot = [cohens_d(net_det[rng.integers(0, n_det, n_det)], net_sta[rng.integers(0, n_sta, n_sta)])
                for _ in range(draws)]
        out["net_displacement"]["cohens_d_interval"] = np.percentile(boot, [2.5, 97.5]).tolist()
        centroids_det = (multiplicity(rng, n_det, draws) @ det.reshape(n_det, -1)).reshape(draws, hours, -1)
        centroids_sta = (multiplicity(rng, n_sta, draws) @ sta.reshape(n_sta, -1)).reshape(draws, hours, -1)
        resampled = space.distance(centroids_det, centroids_sta) / scale
        out["divergence_interval"] = np.percentile(resampled, [2.5, 97.5], axis=0).tolist()
        out["knn_purity_interval"] = {str(k): curve_interval(purity[k], y, rng, draws, balanced=False)
                                      for k in neighbours}
        out["centroid_accuracy_interval"] = curve_interval(centroid_correct, y, rng, draws, balanced=True)
    return out


def direction_curve(a: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    """Direction accuracy at hours 1 ... last, and the angle between the two cohort directions at the last hour."""
    hours = a.shape[1]
    correct = np.zeros((hours, len(a)), dtype=np.float32)
    for h in range(1, hours):
        correct[h] = direction(a, y, h)[0]
    cosine = direction(a, y, hours - 1)[1]
    accuracy = [balanced_accuracy(correct[h], y) for h in range(1, hours)]
    lo, hi = curve_interval(correct, y, rng, draws, balanced=True)

    v = a[:, -1] - a[:, 0]
    cosines = []
    for _ in range(draws):
        det, sta = resample(y, rng)
        cosines.append(float(unit(v[det].mean(0)) @ unit(v[sta].mean(0))))
    cos_lo, cos_hi = np.percentile(cosines, [2.5, 97.5])

    def degrees(c: float) -> float:
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    return {"accuracy": accuracy, "accuracy_interval": [lo[1:], hi[1:]], "cosine": cosine,
            "angle": degrees(cosine), "angle_interval": [degrees(cos_hi), degrees(cos_lo)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    cohorts = json.loads((output_dir / "cohorts.json").read_text())
    stay_ids = [r["stay_id"] for r in cohorts["eligible"]]
    position = {stay_id: i for i, stay_id in enumerate(stay_ids)}

    x, pool = load_trajectories(Path(config["embedding_dir"]) / args.encoder / "test", stay_ids,
                                config["hours"], config["pool"])
    pca = PCA(n_components=config["pca"]["dim"], random_state=config["pca"]["seed"]).fit(pool)
    spaces = {"l2": Space("l2"), "cos": Space("cos"), "pca": Space("pca", pca)}
    transformed = {name: space.transform(x) for name, space in spaces.items()}
    scales = {name: typical_distance(space, space.transform(pool), config["typical_distance"])
              for name, space in spaces.items()}

    boot, encoder_seed = config["bootstrap"], config["encoders"][args.encoder]["bootstrap_seed"]
    rng = np.random.default_rng([boot["seed"], encoder_seed])
    subsample = cohorts["subsample"]["deteriorating"] + cohorts["subsample"]["stable"]
    results = {"pca_explained_variance": float(pca.explained_variance_ratio_.sum()), "configurations": {}}
    for definition, sample, space in CONFIGURATIONS:
        labels = np.array(cohorts["labels"][definition])
        if sample == "all":
            rows = np.flatnonzero(labels >= 0)
            y = labels[rows]
        else:
            rows = np.array([position[s] for s in subsample])
            y = np.array([1] * len(cohorts["subsample"]["deteriorating"]) + [0] * len(cohorts["subsample"]["stable"]))
        first = (definition, sample, space) == CONFIGURATIONS[0]
        results["configurations"]["|".join((definition, sample, space))] = analyse(
            spaces[space], transformed[space][rows], y, scales[space], config["neighbours"],
            rng if first else None, boot["draws"] if first else 0,
        )

    main_labels = np.array(cohorts["labels"]["main"])
    rows = np.flatnonzero(main_labels >= 0)
    results["null"] = []
    for relabelled in cohorts["null_labels"]:
        r = analyse(spaces["l2"], x[rows], np.array(relabelled), scales["l2"], config["neighbours"])
        results["null"].append({
            "cohens_d": r["net_displacement"]["cohens_d"], "divergence_last": r["divergence"][-1],
            "centroid_accuracy_last": r["centroid_accuracy"][-1],
            "direction_accuracy_last": r["direction_accuracy_last"],
            "direction_cosine_last": r["direction_cosine_last"],
        })
    results["direction"] = direction_curve(
        x[rows], main_labels[rows], np.random.default_rng([*boot["direction_seed"], encoder_seed]), boot["draws"],
    )

    output = ensure_dir(output_dir / "metrics") / f"{args.encoder}.json"
    output.write_text(json.dumps(results))
    main_result = results["configurations"]["main|all|l2"]
    logger.info("%s: d %.2f, divergence %.3f, centroid accuracy %.3f, direction accuracy %.3f", args.encoder,
                main_result["net_displacement"]["cohens_d"], main_result["divergence"][-1],
                main_result["centroid_accuracy"][-1], results["direction"]["accuracy"][-1])


if __name__ == "__main__":
    main()
