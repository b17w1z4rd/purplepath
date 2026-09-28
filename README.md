# PurplePath

A machine learning research project for planning **authorized red team exercises**. It ranks high-level ATT&CK scenario categories against measured defensive coverage. It does not run payloads, scan networks, authenticate to targets, or execute scenarios.

PurplePath uses an offline contextual bandit: one scenario is logged per exercise, its probability of selection is recorded, and the outcome is whether the defenders found a meaningful detection gap. A separate linear model estimates the gap rate for each scenario. An uncertainty bonus can prioritize under-tested scenarios. Validation logs choose that bonus, and a chronologically later test set estimates policy value with inverse propensity scoring (IPS) and self-normalized IPS (SNIPS).

## Quick start

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
purplepath generate --out data/synthetic_exercises.csv --episodes 1500
purplepath train --data data/synthetic_exercises.csv --model artifacts/policy.json --report artifacts/evaluation.json
purplepath rank --model artifacts/policy.json --context examples/context.json
python -m unittest discover -s tests -v
```

The committed `artifacts/evaluation.json` is a reproducible synthetic demo report. The current held-out IPS estimate is **0.333** versus **0.217** for a fixed first-scenario policy, with only 74 and 70 matching logged choices respectively. The synthetic generator defines the answer distribution, so these numbers are **not evidence of operational performance**. IPS can be noisy; compare confidence intervals and gather more randomized exercise logs before acting on a small difference.

## Data contract

Each CSV row represents one independently scoped exercise episode, with one logged scenario choice. Required columns:

| Column | Meaning |
| --- | --- |
| `day` | Integer time bucket; earliest 60% train, next 20% validation, latest 20% test |
| `episode_id` | Unique exercise ID; duplicates are rejected |
| `action` | One of the exact five ATT&CK scenario names in `ACTIONS` |
| `propensity` | Known probability that the logging policy chose this action in this context, strictly above zero |
| `gap_found` | 1 if the exercise identified a confirmed detection gap, otherwise 0 |
| `edr_coverage`, `identity_coverage`, `network_coverage`, `logging_maturity`, `asset_criticality` | Pre-exercise context on a documented 0–1 scale |

**Collect propensities when assigning exercises.** Do not infer them after observing outcomes. The demo logger uniformly randomizes over five scenarios, so each recorded propensity is 0.2. If your real program uses a different logger, supply its actual probabilities and ensure every candidate scenario has nonzero selection probability. Train/test windows must represent different times. Keep post-exercise findings out of context fields.

## Method and limitations

For each scenario, ridge regression fits `gap_found` from the context features on historical training episodes. The score is `clip(predicted gap rate + alpha × uncertainty, 0, 1)`. Validation SNIPS selects alpha among 0, 0.25, 0.5, and 1. The final policy is evaluated once on the test period against a fixed T1059 baseline. IPS weights matching logged exercises by `1/propensity`; SNIPS divides by total matching weight. The report gives matching counts and effective sample size to expose weak support.

The exercise categories are deliberately broad. The score indicates **where to inspect defensive coverage**, not a likelihood of successful compromise. Linear predictions are clipped but are not calibrated probabilities. Historical confounding, changing controls, missed labels, and selection bias can invalidate results. A real deployment needs documented scopes, human review, more diverse environments, uncertainty intervals, stronger baselines, and a holdout across organizations.

## Repository layout

- `src/purplepath/bandit.py`: generator, schema checks, chronological split, model, off-policy evaluation, ranking.
- `src/purplepath/cli.py`: `generate`, `train`, and `rank` commands.
- `tests/`: deterministic integration and IPS checks.
- `examples/context.json`: sample coverage context.
- `artifacts/evaluation.json`: reproducible demo output (generated policy JSON is local).

MIT License. See [LICENSE](LICENSE).
