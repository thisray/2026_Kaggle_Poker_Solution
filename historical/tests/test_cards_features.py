import pytest
from pokerlab.cards import parse_cards, board_at_street, rank5, best_rank, heads_up_equity, heads_up_call_ev
from pokerlab.features import pair_outcome_features, is_price_increasing, standardized_binary_residual, require_features
from pokerlab.state import classify_all_in, decision_board, verified_increment


@pytest.mark.parametrize("raw", ["As Kd 10h", "[\"AS\", \"Kd\", \"Th\"]", ["A♠","K♦","T♥"], "AsKdTh"])
def test_parser(raw):
    assert parse_cards(raw)==("As","Kd","Th")


@pytest.mark.parametrize("raw", ["AsAs","ZZ","AsKdX","1h","As Ah As"])
def test_bad_cards(raw):
    with pytest.raises(ValueError): parse_cards(raw)


@pytest.mark.parametrize("street,n", [("preflop",0),("flop",3),("turn",4),("river",5)])
def test_board_prefix(street,n):
    assert len(board_at_street("AsKdQhJcTs",street))==n


def test_board_too_short():
    with pytest.raises(ValueError): board_at_street("AsKdQh","river")


def test_rank_categories():
    hands=["AsKd9h7c3s","AsAh9h7c3s","AsAh9h9c3s","AsAhAc7c3s","AsKdQhJcTs",
           "AsJs9s7s3s","AsAhAc7c7s","AsAhAcAd3s","AsKsQsJsTs"]
    ranks=[rank5(h) for h in hands]
    assert [r[0] for r in ranks]==list(range(9))
    assert ranks==sorted(ranks)


def test_wheel_and_two_trips_full_house():
    assert rank5("As2d3h4c5s")==(4,5)
    assert best_rank("AsAhAcKdKhKc2s")== (6,14,13)


def test_suit_and_input_permutation_invariance():
    a="AsAhAcKdKhKc2s"
    b="AdAcAhKsKcKh2d"
    assert best_rank(a)==best_rank(b)==best_rank(list(reversed(parse_cards(a))))


def test_exact_river_equity_and_ties():
    assert heads_up_equity("AsAh","KdKh","2c3c7h9sJd")["equity"]==1
    assert heads_up_equity("2s3h","4s5h","AsKdQhJcTs")["equity"]==.5


def test_turn_equity_is_exact_and_uses_44_runouts():
    r=heads_up_equity("AsAh","KdKh","2c3c7h9s")
    assert r["exact"] and r["evaluated_runouts"]==44
    assert 0<r["equity"]<1


def test_monte_carlo_reproducible():
    args=("AsAh","KdKh","2c3c7h")
    assert heads_up_equity(*args,samples=8,seed=2,exact_limit=1)==heads_up_equity(*args,samples=8,seed=2,exact_limit=1)


def test_overlap_cards_rejected():
    with pytest.raises(ValueError): heads_up_equity("AsAh","AsKh","2c3c7h")


def test_call_ev():
    assert heads_up_call_ev(.25,30,10)==0
    assert heads_up_call_ev(1,30,10)==30
    assert heads_up_call_ev(0,30,10)==-10
    with pytest.raises(ValueError): heads_up_call_ev(1.1,30,10)


def test_pair_swap_and_stake_scale_invariance():
    a=pair_outcome_features(-40,30,40,20,2)
    assert a==pair_outcome_features(30,-40,20,40,2)
    assert a==pair_outcome_features(-400,300,400,200,20)
    assert a["f_transfer_proxy_max_bb"]==15


@pytest.mark.parametrize("action,increment,cost,result", [("all_in",10,10,False),
    ("all-in",5,10,False),("all_in",11,10,True),("raise",20,10,True),
    ("call",10,10,False),("check",0,0,False),("bet",10,0,True)])
def test_allin_price_change(action,increment,cost,result):
    assert is_price_increasing(action,increment,cost) is result


def test_unknown_action_fails_closed():
    with pytest.raises(ValueError): is_price_increasing("mystery",10,0)


def test_feature_mismatch_cannot_silently_fallback():
    with pytest.raises(KeyError): require_features({"directed_signal":1},["directed_score"])


def test_residual_opportunities():
    assert standardized_binary_residual([0,1],[.5,.5])==0
    with pytest.raises(ValueError): standardized_binary_residual([1],[1.2])


def test_verified_action_state_semantics():
    assert verified_increment(5, 10) == 5
    assert classify_all_in("all-in", 10, 10) == "all_in_call"
    assert classify_all_in("all_in", 11, 10) == "all_in_aggressive"
    assert classify_all_in("call", 10, 10) == "not_all_in"
    assert decision_board("AsKdQhJcTs", "turn") == ("As", "Kd", "Qh", "Jc")
    with pytest.raises(ValueError): verified_increment(11, 10)
