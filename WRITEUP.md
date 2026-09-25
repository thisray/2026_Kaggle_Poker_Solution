# Ranking Suspicious Poker Pairs and Finding Reviewable Hands

## Result and task

Our team, **thisray**, placed provisionally **6th of 370**, with **0.92501 public / 0.92888 private** from `r10_ci.csv`. This writeup describes the competition-time solution and the evidence-retrieval change behind that submission.

The task was retrospective analysis of synthetic poker histories: rank potentially coordinated player pairs, predict a behavior family, and retrieve up to five supporting hands. The score combined **Pair AP (70%)**, **Evidence MAP@5 (20%)**, and **Behavior MAP (10%)**. We separated pair ranking from hand retrieval: a suspicious relationship can contain many ordinary hands, while a striking individual hand can have an innocent explanation.

## Ranking pairs and assigning behavior

We built features from betting actions, amounts, cards, boards, positions, stacks, outcomes, and responses to other players. They described one-way chip movement, passive responses to a partner, pressure on outsiders, and departures from ordinary play in comparable situations. These hand-level observations and their pair-level summaries fed a weighted rank blend of one LightGBM model and two CatBoost models. IDs linked records and defined groups; their encodings were not features.

The public labels were incomplete. We used unlisted pairs as imperfect comparison examples and excluded some model-flagged potential positives from negative training. Supervised validation was grouped by player pool. However, repeated development and shared upstream features meant these diagnostics were not a fully nested evaluation of the entire final system.

A separate classifier handled the three disclosed families: directed transfer, soft play, and coordinated isolation. We also used a gameplay statistic relating a player's actions to the partner's private-card strength to nominate fourth-family candidates. This route assigned **77 evaluation pairs** to `other_coordination` and adjusted their ranking. Those assignments were model-derived hypotheses about the undisclosed family.

## Retrieving supporting hands

The baseline evidence pipeline used family-specific hand scores, additional ranking features, and a blend with **TabICLv2**, an external pretrained tabular model. Family-specific ranking let chip-flow direction influence directed-transfer evidence, passive partner responses influence soft-play evidence, and outsider pressure influence isolation evidence.

The fourth-family candidates had their own evidence rule. Our earlier **NDw** submission favored early shared hands with a model-indicated partner-card-linked decision and a pot won by either pair member. This distinguished a candidate relationship from the particular hands used to support it.

The final r10 change addressed coordinated isolation. Each public positive pair had at most five listed evidence hands. We modeled a full list as potentially censoring later qualifying events: after the fifth listed hand, an unlisted hand should not automatically become a clean training negative.

The event model estimated which hands qualified, then accounted for the probability that fewer than five qualifying events occurred earlier. We blended this ordering with the existing evidence rank and applied gameplay-based eligibility conditions. Chronology referred to recorded hand times, not CSV row order. The first-five assumption was our model of evidence selection, not an organizer-confirmed description of the hidden generator.

The patch was applied to **591 routed pairs**, changing **564 ordered evidence lists**. Pair risks and predicted behaviors remained identical to NDw.

## What the leaderboard showed

| Competition-time submission | Public | Private |
| --- | ---: | ---: |
| NDw base | 0.92303 | 0.92893 |
| r10 with CI evidence patch | 0.92501 | 0.92888 |
| Later r32 ensemble | 0.92456 | 0.92738 |

The CI patch gained **0.00198 public** and changed **−0.00005 private**. With risks and behaviors fixed, this measures the complete evidence patch; it does not isolate the censoring assumption from its other changes. The public improvement did not become a private improvement.

Kaggle automatically evaluated r10 and r32, our two highest-public-score submissions, because we made no manual final selections. NDw therefore did not determine our final team score despite its slightly higher private result. The expanded r32 combination also scored below r10 on both splits.

## Materials and limits

**[Code and execution notes](https://github.com/thisray/2026_Kaggle_Poker_Solution) · [Five evidence case reviews](https://github.com/thisray/2026_Kaggle_Poker_Solution/blob/main/docs/CASE_REVIEWS.md)**

The five reviews use hands actually submitted in r10. Each identifies the pair and hands, describes observable behavior, and gives a plausible benign alternative. Predicted behavior is a review hypothesis; synthetic-benchmark performance does not establish effectiveness on real-player data.

The current implementation is a **partial reconstruction**, not yet a verified reproduction of the selected submissions. Its missing components and execution results are documented separately from the competition-time method above. Competition data are not redistributed.
