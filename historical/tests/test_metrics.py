import numpy as np
import pandas as pd
import pytest
from pokerlab.metrics import (stable_ap, evidence_ap5, clean_evidence, score_submission,
                             EVIDENCE_COLUMNS, SUBMISSION_COLUMNS, KNOWN)
from pokerlab.validation import validate_submission


def frames():
    truth = pd.DataFrame({"pair_id": ["a", "b", "c", "d", "e"], "label": [1, 1, 1, 1, 0],
        "behavior_family": [*KNOWN, "other_coordination", "none"]})
    pred = pd.DataFrame({"pair_id": truth.pair_id, "risk_score": [.9,.8,.7,.6,.1],
                        "predicted_behavior": truth.behavior_family})
    for c in EVIDENCE_COLUMNS:
        pred[c] = "NO_EVIDENCE"
    pred["evidence_hand_1"] = ["ha", "hb", "hc", "hd", "hn"]
    ev = {x: {"h"+x} for x in "abcd"}
    return truth, pred[list(SUBMISSION_COLUMNS)], ev


@pytest.mark.parametrize("y,s,expected", [([1,0,1],[.9,.8,.7],(1+2/3)/2),
    ([0,0],[1,0],0), ([1,1],[0,0],1), ([0,1],[.5,.5],.5), ([1,0],[.5,.5],1)])
def test_stable_ap(y,s,expected):
    assert stable_ap(y,s) == pytest.approx(expected)


@pytest.mark.parametrize("y,s", [([2],[1]), ([1],[np.nan]), ([1,0],[1]), ([[1]],[[1]])])
def test_bad_ap(y,s):
    with pytest.raises(ValueError): stable_ap(y,s)


@pytest.mark.parametrize("pred,truth,result", [(["a","b"],{"a","b"},1),
    (["x","a","b"],{"a","b"},(1/2+2/3)/2),
    (["NO_EVIDENCE","a"],{"a"},1), (["a","a","b"],{"a","b"},(1+2/3)/2),
    (["a"],set(),0), (["x"],{"a"},0), (list("abcde"),set("abcdefg"),1)])
def test_evidence(pred,truth,result):
    assert evidence_ap5(pred,truth) == pytest.approx(result)


def test_appending_unused_slot_cannot_lower_score():
    truth=set("abc")
    for prefix in [[],["a"],["x","b"],["x","a","z"]]:
        for extra in ["a","b","c","d"]:
            if extra not in prefix:
                assert evidence_ap5([*prefix, extra],truth) >= evidence_ap5(prefix,truth)


def test_clean():
    assert clean_evidence([None,np.nan,"", "NO_EVIDENCE"," a "]) == ["a"]


def test_perfect_composite():
    truth,pred,ev = frames()
    assert score_submission(truth,pred,ev).total == pytest.approx(1)


def test_evidence_independent_of_risk_and_behavior():
    truth,pred,ev = frames()
    base = score_submission(truth,pred,ev)
    pred.risk_score = 0.0; pred.predicted_behavior = "none"
    assert score_submission(truth,pred,ev).evidence_map5 == base.evidence_map5 == 1


def test_none_and_other_same_numeric_score():
    truth,pred,ev = frames()
    pred.loc[3,"predicted_behavior"] = "none"
    a=score_submission(truth,pred,ev).as_dict()
    pred.loc[3,"predicted_behavior"] = "other_coordination"
    assert a == score_submission(truth,pred,ev).as_dict()


def test_empty_positive_evidence_counts_zero():
    truth,pred,ev=frames(); del ev["d"]
    assert score_submission(truth,pred,ev).evidence_map5 == pytest.approx(.75)


def test_absent_families_still_three_way_average():
    truth,pred,ev=frames(); truth=truth.iloc[:1]; pred=pred.iloc[:1]
    assert score_submission(truth,pred,ev).behavior_map == pytest.approx(1/3)


def test_pair_id_tie_order_and_submission_row_order():
    truth,pred,ev=frames(); pred.risk_score=.5
    a=score_submission(truth,pred,ev).total
    assert score_submission(truth.iloc[::-1], pred.sample(frac=1,random_state=8),ev).total == a


@pytest.mark.parametrize("change", ["duplicate","nan","blank","risk_inf","risk_negative","risk_large","bad_family","extra_pair","missing_pair","columns"])
def test_strict_submission_rejects(change):
    truth,pred,ev=frames()
    if change=="duplicate": pred.loc[0,"evidence_hand_2"]="ha"
    elif change=="nan": pred.loc[0,"evidence_hand_1"]=None
    elif change=="blank": pred.loc[0,"evidence_hand_2"]=" "
    elif change=="risk_inf": pred.loc[0,"risk_score"]=np.inf
    elif change=="risk_negative": pred.loc[0,"risk_score"]=-.1
    elif change=="risk_large": pred.loc[0,"risk_score"]=1.1
    elif change=="bad_family": pred.loc[0,"predicted_behavior"]="unknown"
    elif change=="extra_pair": pred.loc[0,"pair_id"]="extra"
    elif change=="missing_pair": pred=pred.iloc[:-1]
    elif change=="columns": pred=pred[pred.columns[::-1]]
    with pytest.raises(ValueError): validate_submission(pred,truth.pair_id)


def test_membership_and_period():
    truth,pred,ev=frames()
    legal=pd.DataFrame({"pair_id":list("abcde"),"hand_id":["ha","hb","hc","hd","hn"],"phase":"evaluation"})
    assert validate_submission(pred,truth.pair_id,legal)["membership_checked"]==1
    legal.loc[0,"phase"]="development"
    with pytest.raises(ValueError): validate_submission(pred,truth.pair_id,legal)


def test_duplicate_pair_rejected():
    truth,pred,ev=frames(); pred.loc[0,"pair_id"]="b"
    with pytest.raises(ValueError): validate_submission(pred,truth.pair_id)
