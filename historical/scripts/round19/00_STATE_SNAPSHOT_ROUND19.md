# Poker Round 19 State Snapshot

查核時間：2026-09-19 15:48 UTC
Canonical source before this round：origin/main 8dc10e4
Competition：detect-suspicious-value-transfers-in-poker

## Verified live numbers

- Live Top-1 public：0.93876。
- Team best public：0.92303，ref 56343238，candidate r2n_NDw_on_r2j2m.csv。
- Public gap：0.01573；team live rank 約 #9。
- Deadline：2026-09-20 22:00 UTC。
- Regular quota audit：2026-09-19 used 5/5，remaining 0；overall thisray=17。
- Next UTC reset：2026-09-20 00:00 UTC；若規則不變，reset 後最多 5 次 regular submissions。
- Private scores：目前官方 submission history 尚未提供。
- This round：沒有 submission、沒有大型運算、沒有 raw data／weights 複製到 M4。

## Candidate queue and checksums

Current R5 candidates are validated on GB10 and are not copied into this M4 handoff. The repository records SHA-256 prefixes; full candidate manifests remain at the GB10 artifact path.

| Candidate | Role | Evidence rule | Other count | Recorded SHA-256 prefix |
|---|---|---|---:|---|
| r5_subh | S1 | ACT | 91 | 48d3324ee89d88c3 |
| r5_subp | S2 | PLANT | 91 | 71312ed684f05141 |
| r5_mix | hedge | ACT + PLANT | 91 | 8a158469ba24eb52 |
| r5_witness | alternative | WITNESS | 91 | ba8cad47ee825022 |
| r5_plus_subh | later only | ACT + B7 | 94 | 920a8143a0200346 |
| r5_plus_subp | later only | PLANT + B7 | 94 | 90481d5d6ecb9330 |
| r5_plus_mix | later only | mixed + B7 | 94 | 0ca16d7c6736b27a |
| r5_plus_witness | later only | WITNESS + B7 | 94 | f5e982541727210f |

## Round 19 highest-value handoff

1. Pre-peer plus matched conditional source null: separate partner private-card source from public-action mediator and unconditioned outsider/card-strength confounding.
2. Observable-witness coarsening decoder: evidence must contain a behavior-specific visible action; model capped partial order instead of treating all unlisted hands as negatives.
3. Temporal self-baseline and gated adaptation: compare the same pair's eval residual with its dev-period benign baseline, then test clipped target calibration.
4. Directional two-state HMM/HSMM: model episodic A→B/B→A source states and decode evidence marks, initially without changing risk.
5. Pre-peer transportable regret and ledger-constrained flow remain later probes, gated by negative controls.

## Open review questions

- Does partner-vs-matched-outsider separation remain after pre-peer, partner-fold, seat and time permutation controls?
- Does observable-witness masking preserve candidate Recall@32 while improving nested MAP@5?
- Is temporal adaptation stable under pool-held-out forward validation with acceptable ESS?
- Does a hidden episode state survive time-shuffle and phase-matched nulls?
- Which R5 evidence rule, if any, is robust enough to spend reset-day quota? No local estimate is an LB result.
