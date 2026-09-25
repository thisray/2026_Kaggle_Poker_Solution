# Codex task: independent code review + synthetic tests of the R9 fourth-family evidence decoder (testing only)

Work ONLY inside this directory (`codex_review_r9/`). Do NOT modify any file outside it. Do NOT submit anything to Kaggle, do NOT use the network.
Python: use `/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python` (conda env; never system python). Limit CPU: prefix heavy commands with `taskset -c 12,14 nice -n 12` and set `OPENBLAS_NUM_THREADS=1`.

## Files under review (read-only, in the parent directory `..`)
- `../t37_re2d.py` — pair random-effects model and first-5 decoders (the file used to build the R9 candidates with env `BASE=r5r_subh.csv PREFIX=r9 SKIPCV=1 WIT=1 OOT=1 RV1=1`).
- `../f4_random_effects.py` — R19 helper (`quadrature`, `first_k_at_nodes`).
- `../t40_v1_ratio.py` — policy_v1 likelihood ratios for decisions that policy_v2 trained on.

## Intended model (check the code implements exactly this)
Per pair p, hands h (time-ordered), member decisions d in h with likelihood ratio r_d = P(action | partner's cards) / P(action | own cards), street pre/post.
- Z_h ~ Bernoulli(rho_p); rho_p ~ Beta(mu*kappa, (1-mu)*kappa).
- Given Z_h = 1, decision d uses partner cards with prob s_d = sigmoid(logit(s_street) + sigma*u_p), u_p ~ N(0,1).
- Hand likelihood ratio vs normal play: Z_h=0 -> 1; Z_h=1 -> prod_d (1 - s_d + s_d r_d). Pair marginal = integral over rho and u of prod_h [(1-rho) + rho * exp(lf_h)], lf_h = sum_d log(1 - s_d + s_d r_d).
- Models: M0 fixed rho, fixed s; M1 random rho; M2 random u; M3 both. Quadrature: Gauss-Jacobi for rho (scipy roots_jacobi(n, alpha=(1-mu)kappa-1, beta=mu kappa-1), rho=(x+1)/2), Gauss-Hermite-e for u.
- Per hand at each node: q_plant = sigmoid(logit rho + lf_h); q_act = q_plant * (1 - exp(l0_h - lf_h)) with l0_h = sum_d log(1 - s_d); q_wit = q_plant * (1 - prod_d (1 - u_d)), u_d = s max(r-1,0)/(1-s+s r).
- Decoder: at every node, first-5 DP over the full time-ordered hand sequence of q * pairwin (probability that hand h is an event AND fewer than 5 events happened before it); THEN average over the pair posterior node weights (E[DP(q)], not DP(E[q])). Tie-break adds 1e-6 * the same DP without the pair-win factor.
- OOT/RV1: rows whose decision k is in policy_v2's training sample RandomState(1).choice(N, 6_000_000, replace=False) get r := r_v1 (from t40) if k is NOT in policy_v1's sample RandomState(0).choice(N, 4_000_000, replace=False), else r := 1 (uninformative; the decision still counts in l0 for the ACT posterior).

## Tasks
1. Code review of the three files against the intended model: indexing/alignment (row order, hand grouping by (slot, h), pair grouping, the `keep` mask on nodes, u-major reshape order, merges on (k, slot)), numerical issues, and anything that would make the written candidate evidence differ from the intended model. Report each finding with file:line, severity (bug / risk / nit) and a concrete fix.
2. Synthetic tests (write them as `test_*.py` here and run them):
   a. `first_k_at_nodes` vs brute-force enumeration over all 2^n event patterns (n <= 10, several node columns, random q).
   b. Beta quadrature: E[rho], E[rho^2] from `quadrature`/roots_jacobi weights vs analytic Beta moments for (mu, kappa) in {(0.3, 2), (0.75, 4.9), (0.9, 10)}.
   c. Re-implement `pair_ll` for M3 with explicit Python loops (nodes x hands x decisions) on a small random dataset (3 pairs, 5 hands each, 1-4 decisions per hand, random r and street) and compare with t37's vectorised `pair_ll` (import the functions by exec-ing only the function definitions, or copy them verbatim into the test).
   d. Same for the decoder: for one small random pair, compute the integrated first-5 scores by explicit loops and compare with t37's vectorised result.
3. Write `REVIEW.md` here (English is fine): summary verdict (are the R9 evidence picks trustworthy?), findings table, test results with the exact commands you ran.
