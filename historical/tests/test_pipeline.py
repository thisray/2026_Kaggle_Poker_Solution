from dataclasses import replace
import numpy as np
import pytest
from pokerlab.cli import toy_data
from pokerlab.baseline import Config, fit_models, predict, run_group_cv
from pokerlab.validation import validate_submission
from pokerlab.metrics import EVIDENCE_COLUMNS


def config():
    return Config(tuple(f"f_signal_{j}" for j in range(3)), ("f_hand_signal",), n_splits=3,max_iter=10,min_samples_leaf=5)


def test_group_cv_end_to_end():
    p,l,h,e=toy_data()
    oof,report=run_group_cv(p,l,h,e,config())
    validate_submission(oof,p.pair_id,h,phase="development")
    lookup={r["pair_id"]:r["fold"] for r in report["fold_manifest"]}
    for _,g in p.groupby("table_id"):
        assert len({lookup[x] for x in g.pair_id})==1
    assert len(report["folds"])==3 and 0<=report["pooled_oof"]["total"]<=1
    # It is a software fixture. Do NOT assert competition performance thresholds.


def test_all_pairs_get_evidence_even_at_zero_risk():
    p,l,h,e=toy_data()
    models=fit_models(p,l,h,e,replace(config(),risk_threshold=1))
    pred=predict(models,p,h)
    assert pred.predicted_behavior.eq("none").all()
    assert pred[list(EVIDENCE_COLUMNS)].ne("NO_EVIDENCE").all().all()


def test_unknown_feature_and_eval_cv_rejected():
    p,l,h,e=toy_data()
    with pytest.raises(ValueError): fit_models(p,l,h,e,replace(config(),pair_features=("pair_id",)))
    p.phase="evaluation";h.phase="evaluation"
    with pytest.raises(ValueError): run_group_cv(p,l,h,e,config())


def test_predict_no_hands_fills_sentinel():
    p,l,h,e=toy_data()
    models=fit_models(p,l,h,e,config())
    pred=predict(models,p,h.iloc[:0])
    assert pred[list(EVIDENCE_COLUMNS)].eq("NO_EVIDENCE").all().all()
