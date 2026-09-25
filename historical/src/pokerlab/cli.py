from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import platform
import uuid
import numpy as np
import pandas as pd
import sklearn
from .baseline import Config, run_group_cv, fit_models, predict
from .metrics import KNOWN
from .raw_audit import audit
from .raw_features import FeatureBuildConfig, audit_feature_membership, build_features
from .advanced_experiments import AdvancedConfig, run_advanced
from .train import TrainConfig, train_and_submit


def toy_data():
    """Human-made software fixture, with no real competition data or labels."""
    rng = np.random.default_rng(2026)
    pairs, labels, hands, evidence = [], [], [], {}
    for pool in range(12):
        for k in range(16):
            pid = str(uuid.UUID(int=int(rng.integers(1, 2**62))))
            kind = k if k < 3 else -1
            signal = rng.normal(0, 0.7, 3)
            if kind >= 0:
                signal[kind] += 3.3
            pairs.append({"pair_id": pid, "table_id": f"toy_pool_{pool}", "phase": "development",
                          **{f"f_signal_{j}": float(signal[j]) for j in range(3)}})
            labels.append({"pair_id": pid, "label": int(kind >= 0),
                           "behavior_family": KNOWN[kind] if kind >= 0 else "none"})
            chosen = set()
            for j in range(10):
                hand = str(uuid.UUID(int=int(rng.integers(1, 2**62))))
                planted = kind >= 0 and j < 3
                hands.append({"pair_id": pid, "hand_id": hand, "phase": "development",
                              "f_hand_signal": float(rng.normal(3 if planted else 0, .7))})
                if planted:
                    chosen.add(hand)
            evidence[pid] = chosen
    return pd.DataFrame(pairs), pd.DataFrame(labels), pd.DataFrame(hands), evidence


def demo(outdir: str):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pairs, labels, hands, ev = toy_data()
    config = Config(tuple(f"f_signal_{j}" for j in range(3)), ("f_hand_signal",),
                    n_splits=3, max_iter=40, min_samples_leaf=8)
    oof, report = run_group_cv(pairs, labels, hands, ev, config)
    report["dataset"] = "SOFTWARE_FIXTURE_ONLY_NOT_KAGGLE_COMPETITION"
    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "sklearn": sklearn.__version__}
    oof.to_csv(out/"toy_oof.csv", index=False)
    (out/"toy_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame(report["fold_manifest"]).to_csv(out/"toy_fold_manifest.csv", index=False)
    print(json.dumps({"dataset": report["dataset"], "pairs": len(pairs), "hands": len(hands),
                      "output": str(out)}, ensure_ascii=False))


def load_evidence(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, dtype={"pair_id": str, "hand_id": str})
    if not {"pair_id", "hand_id"}.issubset(frame.columns):
        raise ValueError("Evidence CSV requires pair_id,hand_id")
    return frame.groupby("pair_id").hand_id.agg(set).to_dict()


def main():
    parser = argparse.ArgumentParser(description="Research reference utilities; no raw-to-winning-model claim")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo"); p.add_argument("--out", required=True)
    p = sub.add_parser("audit"); p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("build-features")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--memory-limit", default="80GB")
    p = sub.add_parser("audit-membership")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True)
    p = sub.add_parser("train-real")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--max-iter", type=int, default=280)
    p.add_argument("--max-leaf-nodes", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=10)
    p.add_argument("--extra-trees", type=int, default=320)
    p.add_argument("--n-jobs", type=int, default=8)
    p = sub.add_parser("advanced-real")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--max-iter", type=int, default=160)
    p.add_argument("--max-leaf-nodes", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=8)
    p = sub.add_parser("fit-features")
    for name in ("pairs", "labels", "hands", "evidence", "config", "out"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--test-pairs"); p.add_argument("--test-hands")
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.out)
    elif args.command == "audit":
        audit(args.data_dir, args.out)
    elif args.command == "build-features":
        report = build_features(FeatureBuildConfig(
            data_dir=args.data_dir,
            output_dir=args.out,
            threads=args.threads,
            memory_limit=args.memory_limit,
        ))
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif args.command == "audit-membership":
        report = audit_feature_membership(args.data_dir, args.features)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif args.command == "train-real":
        report = train_and_submit(TrainConfig(
            data_dir=args.data_dir,
            features_dir=args.features,
            output_dir=args.out,
            n_splits=args.splits,
            seed=args.seed,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            extra_trees=args.extra_trees,
            n_jobs=args.n_jobs,
        ))
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif args.command == "advanced-real":
        report = run_advanced(AdvancedConfig(
            data_dir=args.data_dir,
            features_dir=args.features,
            output_dir=args.out,
            n_splits=args.splits,
            seed=args.seed,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
        ))
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        if bool(args.test_pairs) != bool(args.test_hands):
            parser.error("--test-pairs and --test-hands must be provided together")
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        pairs, labels, hands = [pd.read_csv(getattr(args, n), dtype={"pair_id": str, "hand_id": str})
                               for n in ("pairs", "labels", "hands")]
        cfg = json.loads(Path(args.config).read_text())
        cfg["pair_features"], cfg["hand_features"] = tuple(cfg["pair_features"]), tuple(cfg["hand_features"])
        config = Config(**cfg)
        ev = load_evidence(args.evidence)
        oof, report = run_group_cv(pairs, labels, hands, ev, config)
        oof.to_csv(out/"oof.csv", index=False)
        (out/"cv_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        models = fit_models(pairs, labels, hands, ev, config)
        with (out/"reference_models.pkl").open("wb") as f:
            pickle.dump(models, f)
        if args.test_pairs:
            test_pairs = pd.read_csv(args.test_pairs, dtype={"pair_id": str})
            test_hands = pd.read_csv(args.test_hands, dtype={"pair_id": str, "hand_id": str})
            if not test_pairs.phase.eq("evaluation").all():
                raise ValueError("Submission test pairs must have evaluation phase")
            predict(models, test_pairs, test_hands).to_csv(out/"submission.csv", index=False)
        print(f"Saved reference run to {out}; provenance and official parity remain required")

if __name__ == "__main__":
    main()
