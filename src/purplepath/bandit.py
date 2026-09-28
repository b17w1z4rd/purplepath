"""Offline exercise prioritization. This package never runs an emulation or attack."""
import csv
import json
from pathlib import Path

import numpy as np

ACTIONS = ("T1059 Command and Scripting Interpreter",
           "T1078 Valid Accounts", "T1047 Windows Management Instrumentation",
           "T1105 Ingress Tool Transfer", "T1562.001 Disable or Modify Tools")
FEATURES = ("edr_coverage", "identity_coverage", "network_coverage",
            "logging_maturity", "asset_criticality")
COLUMNS = ("day", "episode_id", "action", "propensity", "gap_found", *FEATURES)


def generate(path, episodes=1500, seed=42):
    """Generate one logged, randomized exercise choice per simulated episode."""
    if episodes < 100:
        raise ValueError("At least 100 episodes are needed")
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.array([[-1.2, -.8, -.5, -.8, .2],
                        [-.1, -1.5, -.1, -.5, .2],
                        [-1.1, -.1, -.1, -.8, .2],
                        [-.2, -.1, -1.4, -.7, .2],
                        [-1.5, -.1, -.2, -.5, .2]])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(episodes):
            context = rng.beta(2, 2, size=len(FEATURES))
            # The logger explores every scenario equally. Its known choice
            # probability supports unbiased off-policy estimation.
            action = int(rng.integers(len(ACTIONS)))
            logit = -.15 + np.dot(weights[action], context) + rng.normal(0, .12)
            chance = 1 / (1 + np.exp(-logit))
            outcome = int(rng.random() < chance)
            writer.writerow(dict(zip(COLUMNS, (i // 10, f"lab-{i:06d}", ACTIONS[action],
                                                1 / len(ACTIONS), outcome,
                                                *(round(float(v), 6) for v in context)))))
    return path


def read_data(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty data")
    try:
        day = np.array([int(r["day"]) for r in rows])
        episode = [r["episode_id"] for r in rows]
        action = np.array([ACTIONS.index(r["action"]) for r in rows])
        propensity = np.array([float(r["propensity"]) for r in rows])
        reward = np.array([int(r["gap_found"]) for r in rows])
        context = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid row value or action") from exc
    if len(set(episode)) != len(episode):
        raise ValueError("Each episode must have exactly one logged choice")
    if (not np.isfinite(context).all() or not np.isfinite(propensity).all()
            or np.any((context < 0) | (context > 1))
            or np.any((propensity <= 0) | (propensity > 1))
            or not set(reward).issubset({0, 1})):
        raise ValueError("Features must be in [0,1], reward binary, propensity in (0,1]")
    if len(set(day)) < 3:
        raise ValueError("Need at least three distinct days for chronological splits")
    return day, action, propensity, reward, context


def chronological_split(day):
    unique = np.unique(day)
    a, b = max(1, int(len(unique) * .6)), max(2, int(len(unique) * .8))
    if b >= len(unique) or a >= b:
        raise ValueError("Need more time periods")
    train = np.flatnonzero(day <= unique[a-1])
    val = np.flatnonzero((day > unique[a-1]) & (day <= unique[b-1]))
    test = np.flatnonzero(day > unique[b-1])
    return train, val, test


def design(context):
    x = np.asarray(context, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError("Expected one row per context and five features")
    return np.column_stack([np.ones(len(x)), x])


def fit(context, action, reward, ridge=1.):
    x = design(context)
    matrices, vectors = [], []
    for k in range(len(ACTIONS)):
        chosen = x[action == k]
        if len(chosen) < 5:
            raise ValueError(f"Too few training examples for {ACTIONS[k]}")
        a = chosen.T @ chosen + ridge * np.eye(x.shape[1])
        b = chosen.T @ reward[action == k]
        matrices.append(np.linalg.inv(a))
        vectors.append(np.linalg.solve(a, b))
    return np.stack(vectors), np.stack(matrices)


def predict(context, weights, inverses, alpha=0.):
    x = design(context)
    mean = np.clip(x @ weights.T, 0, 1)
    uncertainty = np.sqrt(np.maximum(0, np.einsum("ni,aij,nj->na", x, inverses, x)))
    return mean, np.clip(mean + alpha * uncertainty, 0, 1), uncertainty


def off_policy_evaluate(policy, actions, propensities, rewards):
    """IPS and SNIPS for a deterministic policy on held-out logged decisions."""
    match = policy == actions
    weights = match / propensities
    ips = np.mean(weights * rewards)
    denominator = np.sum(weights)
    return {"ips": float(ips), "snips": float(np.sum(weights * rewards) / denominator)
            if denominator else None, "matching_rows": int(match.sum()),
            "total_rows": len(policy), "effective_sample_size":
            float(denominator**2 / np.sum(weights**2)) if denominator else 0.}


def train(data, model, report, ridge=1.):
    day, actions, propensities, rewards, context = read_data(data)
    tr, va, te = chronological_split(day)
    weights, inverses = fit(context[tr], actions[tr], rewards[tr], ridge)
    # Explore uncertainty in the next suggested exercise. Tune using
    # validation logs only; test logs are reserved for one final estimate.
    candidates = [0., .25, .5, 1.]
    validations = []
    for alpha in candidates:
        _, ranking, _ = predict(context[va], weights, inverses, alpha)
        evaluation = off_policy_evaluate(np.argmax(ranking, axis=1), actions[va],
                                         propensities[va], rewards[va])
        validations.append((evaluation["snips"] or -1, -alpha, alpha, evaluation))
    _, _, alpha, validation = max(validations)
    _, ranking, _ = predict(context[te], weights, inverses, alpha)
    test = off_policy_evaluate(np.argmax(ranking, axis=1), actions[te],
                               propensities[te], rewards[te])
    baseline = off_policy_evaluate(np.zeros(len(te), dtype=int), actions[te],
                                   propensities[te], rewards[te])
    result = {"data": str(data), "synthetic_data_warning": "Synthetic exercise outcomes are not operational evidence",
              "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
              "ridge": ridge, "chosen_alpha": alpha, "validation": validation,
              "test_policy": test, "test_fixed_scenario_baseline": baseline,
              "observed_test_gap_rate": float(np.mean(rewards[te]))}
    Path(model).parent.mkdir(parents=True, exist_ok=True)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(model).write_text(json.dumps({"actions": ACTIONS, "features": FEATURES,
                                       "weights": weights.tolist(), "inverses": inverses.tolist(),
                                       "alpha": alpha}, indent=2) + "\n")
    Path(report).write_text(json.dumps(result, indent=2) + "\n")
    return result


def rank(model, context):
    saved = json.loads(Path(model).read_text())
    if saved["actions"] != list(ACTIONS) or saved["features"] != list(FEATURES):
        raise ValueError("Incompatible model schema")
    try:
        values = np.array([[float(context[k]) for k in FEATURES]])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Expected five numeric coverage features") from exc
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Coverage values must be finite in [0,1]")
    mean, ranking, uncertainty = predict(values, np.array(saved["weights"]),
                                          np.array(saved["inverses"]), saved["alpha"])
    order = np.argsort(-ranking[0], kind="stable")
    return [{"scenario": ACTIONS[i], "estimated_gap_rate": float(mean[0, i]),
             "uncertainty": float(uncertainty[0, i]), "priority_score": float(ranking[0, i])}
            for i in order]
