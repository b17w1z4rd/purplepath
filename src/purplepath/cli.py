import argparse
import json
from .bandit import generate, rank, train


def main():
    p = argparse.ArgumentParser(description="Offline red team exercise planning")
    s = p.add_subparsers(dest="command", required=True)
    g = s.add_parser("generate")
    g.add_argument("--out", default="data/synthetic_exercises.csv")
    g.add_argument("--episodes", type=int, default=1500)
    g.add_argument("--seed", type=int, default=42)
    t = s.add_parser("train")
    t.add_argument("--data", default="data/synthetic_exercises.csv")
    t.add_argument("--model", default="artifacts/policy.json")
    t.add_argument("--report", default="artifacts/evaluation.json")
    t.add_argument("--ridge", type=float, default=1.)
    r = s.add_parser("rank")
    r.add_argument("--model", default="artifacts/policy.json")
    r.add_argument("--context", required=True)
    args = p.parse_args()
    if args.command == "generate":
        print(generate(args.out, args.episodes, args.seed))
    elif args.command == "train":
        if args.ridge <= 0:
            p.error("--ridge must be positive")
        print(json.dumps(train(args.data, args.model, args.report, args.ridge), indent=2))
    else:
        with open(args.context) as f:
            context = json.load(f)
        print(json.dumps(rank(args.model, context), indent=2))

if __name__ == "__main__":
    main()
