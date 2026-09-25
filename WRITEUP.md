# Ranking Suspicious Poker Pairs and Finding Reviewable Hands

## Result and task

Our team, **thisray**, placed provisionally **6th of 370**, with **0.92501 public / 0.92888 private** from `r10_ci.csv`. The task was to rank suspicious player pairs, predict a coordination family, and retrieve up to five hands supporting each alert. The metric combined **Pair AP (70%)**, **Evidence MAP@5 (20%)**, and **Behavior MAP (10%)**.

We treated pair detection and evidence retrieval as related but separate problems. A coordinated relationship can contain many ordinary hands, while an unusual individual hand can still have a benign explanation. This separation became especially important because evidence retrieval was a meaningful part of the score rather than only an explanation layer.

## Pair ranking and behavior routing

All predictive signals came from published poker data: betting actions and amounts, cards and boards, position, stack and pot context, outcomes, and responses to other players. IDs were used for joins, grouping, folds, and stable bookkeeping; their encodings were not predictive features.

The public labels were positive-unlabelled. Pairs absent from `development_labels.csv` were therefore not treated as ground-truth negatives. We used them as imperfect comparison examples, while excluding some model-flagged likely positives from negative training. Validation was grouped by player pool so folds did not share the same table population.

The main pair-risk score was a rank blend of one LightGBM model and two CatBoost models with weights **0.50 / 0.25 / 0.25**. A separate model routed the three disclosed families: directed transfer, soft play, and coordinated isolation.

We also searched for the undisclosed fourth family using gameplay rather than identifiers. A partner-card-dependence statistic measured whether a player's decisions changed with the partner's private-card strength. This route nominated **77 evaluation pairs** as `other_coordination` and inserted them into the risk ranking. These were model-derived hypotheses about the hidden family, not access to hidden labels.

## Evidence retrieval

Evidence ranking was family-aware. Directed-transfer candidates emphasized directional chip flow and concessions to the partner; soft-play candidates emphasized unusually passive responses to partner aggression; coordinated-isolation candidates emphasized pressure on outsiders together with reduced partner conflict.

The competition-time retrieval stack combined family-specific candidate models, a sequence-neural score, a learned within-pair ranker, and a **24-feature TabICLv2** view. The TabICLv2 features combined eleven candidate/ranking scores with thirteen gameplay-context features, and its within-pair rank was blended with the learned ranker. This gave the evidence system a different objective from pair risk: it had to find a small number of reviewable hands inside an already suspicious relationship.

The fourth-family candidates used a separate **NDw** rule. It favored early shared hands containing a model-indicated partner-card-linked decision and a pot won by either pair member. This kept the relationship-level fourth-family detector separate from the rule selecting particular supporting hands.

## The final r10 change: censored evidence lists

Our last r10 change focused on coordinated isolation. Each public positive pair had at most five listed evidence hands. Treating every unlisted hand as a clean negative is unsafe when later qualifying events may simply be omitted after the list is full.

We therefore modeled a full list as a potentially right-censored event sequence. The event model estimated which hands qualified and then accounted for the probability that fewer than five qualifying events had occurred earlier. We blended this ordering with the existing evidence rank and applied gameplay-based eligibility conditions.

Chronology used the recorded hand time, never CSV row order. The first-five interpretation was our statistical model of the released evidence lists, not an organizer-confirmed description of hidden generator internals.

The CI patch was applied to **591 routed pairs** and changed **564 ordered five-hand lists**. It did not alter pair risks or predicted behaviors.

## A later r32 variant

A later second selected entry, r32, broadened pair-risk ensembling and used larger family-specific evidence ensembles. The competition-time risk fusion combined a 64-model historical group with five exposure-matched models, while later evidence stages used family-specific model zoos before the final routed patches. Kaggle automatically evaluated r10 and r32 because we had not manually selected final entries.

| Competition-time submission | Public | Private |
| --- | ---: | ---: |
| NDw base | 0.92303 | 0.92893 |
| r10 with CI evidence patch | 0.92501 | 0.92888 |
| Later r32 ensemble | 0.92456 | 0.92738 |

The r10 CI patch improved public score by **0.00198** and changed private score by **−0.00005**. Since risk and behavior were fixed, this measures the complete evidence patch, but it does not isolate the censoring assumption from every other detail of that patch. The public improvement did not generalize into a private improvement. The broader r32 entry also scored below r10 on both splits.

## Validation, limitations, and reproduction

Our diagnostics were useful for model selection but were not a fully nested end-to-end evaluation: development labels and upstream representations were reused across repeated experiments. Public feedback also influenced late-stage choices. The benchmark is fully synthetic, so a high model score should be interpreted as a prompt for review, not as evidence about real-player intent.

The accompanying repository contains setup and execution instructions, training and inference code for the selected-submission method, five evidence case reviews, and a separate exact final-assembly audit path. The external TabICLv2 dependency is publicly obtainable and documented. The exact assembly audit reproduced both original selected CSVs; the revised raw-data path has not completed a fresh end-to-end run, so no fresh score or output-similarity claim is made.

**[Code and execution notes](https://github.com/thisray/2026_Kaggle_Poker_Solution) · [Five evidence case reviews](https://github.com/thisray/2026_Kaggle_Poker_Solution/blob/main/docs/CASE_REVIEWS.md)**
