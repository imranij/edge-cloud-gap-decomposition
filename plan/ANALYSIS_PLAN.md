# Paper A: frozen analysis plan

**Status.** Written on 11 October 2026 before any model was fitted on a confirmatory dataset.
Pilot results on SECOM and MetroPT-3 generated the hypotheses, so those two datasets are
excluded from every confirmatory test and reported only as pilots. Dataset agents prepared
the arrays without training or scoring any classifier. Each dataset's edge-node definition is
recorded in Addendum A before its first model run. Every later change is logged in
`plan/DEVIATIONS.md` with its date, its reason, and whether any confirmatory result had been
seen. The SHA-256 of this file is recorded in `plan/FREEZE.txt`.

## 1. Question and estimands

The study covers a two-tier system for binary fault detection. The edge model sees only the
sensors at one physical node. The cloud model sees all sensors. A router decides which windows
the edge escalates to the cloud. For any escalation set E,

  BA(E) = BA(edge) + sum over i in E of g_i,  where g_i = w_i (1[cloud right] - 1[edge right]).

Here w_i = 1/(2 n_pos) for positives and 1/(2 n_neg) for negatives. The study estimates:

- the **model step**: a cloud-family model on the edge view against the edge-family model;
- the **information step**: all sensors against edge sensors only;
- the **combination step**: a stacker on both outputs against the cloud;
- **unresolved disagreement**: the per-window oracle against the stacker;
- **weighted concordance C** of each router with g;
- **capture**, the router's gain over random as a share of the oracle ranking's gain over
  random, summed over escalation budgets from 5% to 30%.

## 2. Datasets and views

There are four confirmatory datasets: the UCI hydraulic rig (`hydraulic`), SKAB (`skab`), the
SEU gearbox (`seu`) and Petrobras 3W real instances (`threew`). Their edge and cloud views
follow one rule.

- **Edge view.** The sensors physically co-located at one node, chosen from documentation
  alone.
- **Cloud view.** All usable sensors. Edge features are a column subset of cloud features,
  because they come from the same shared extractor (`src/escalation/features.py`).

A dataset is dropped only if its check script fails. That would be a deviation.

## 3. Splits

The design is repeated stratified grouped K-fold cross-validation over the dataset's
independent units (group_id).

- **Outer folds.** K = min(5, number of groups containing positives, number of groups
  containing negatives), with K >= 3. R = ceil(20 / K) repeats for tabular pairs (J = K·R ≥ 20
  splits) and R = ceil(10 / K) for the CNN pair. Repeat r uses StratifiedGroupKFold(K,
  shuffle=True, random_state=1000+r), stratified on window labels.
- **Inner split.** The training groups are split once by StratifiedGroupKFold(K_in,
  shuffle=True, random_state=2000+split), with K_in = min(4, groups per class in training).
  One inner fold is the validation set; the rest is the fit set.
- **What uses which set.** Models are trained on the fit set. Thresholds, stackers and
  validation-trained routers use the validation set. Everything is evaluated on the outer test
  fold.
- **Corrected tests.** The Nadeau-Bengio correction uses n_train = fit + validation size and
  n_test = test size, averaged over splits.

## 4. Models

Each pair has an edge family and a cloud family. Each family is trained on each view, which
gives four factorial models per split.

- **Pair A (tabular, primary).**
  - Edge family: logistic regression (standardised, L2, C = 1, class-balanced).
  - Cloud family: balanced random forest (300 trees, min_samples_leaf = 2,
    max_features = sqrt, sampling_strategy = all, replacement = True).
- **Pair B (tabular).**
  - Edge family: small gradient boosting (HistGradientBoosting, max_iter = 50, max_depth = 3,
    learning_rate = 0.1).
  - Cloud family: large gradient boosting (max_iter = 400, max_leaf_nodes = 31,
    learning_rate = 0.05, l2 = 1).
  - Both use class-balanced sample weights.
- **Pair C (raw windows; every dataset that provides raw arrays).**
  - Edge family: a tiny 1D-CNN (two conv layers with 8 and 16 filters, stride 2, global average
    pooling, linear head).
  - Cloud family: a larger 1D-CNN (four conv blocks with 32, 32, 64 and 64 filters,
    batch-norm, stride 2, global average pooling, dropout 0.2).
  - Training: Adam (lr 1e-3, batch 128), class-balanced cross-entropy, at most 30 epochs,
    early stopping on validation balanced accuracy (patience 5), and per-channel
    standardisation from the fit set.
- **Deployed pair.** The edge is (edge family, edge view) and the cloud is (cloud family,
  cloud view).
- **Upgraded edge.** (cloud family, edge view).
- **Self-oracle null (pair A).** A second (cloud family, cloud view) model with seed + 7919.
- **Thresholds.** Every model's threshold maximises balanced accuracy on the validation set.
  This is the primary rule. The composite rule of the original paper (BA + 0.25 F1 + 0.10
  recall) is a sensitivity analysis.
- **Seeds.** Model seed = 3000 + split index.

## 5. Escalation quantities per split (test fold)

- **Balanced accuracy.** BA of all four factorial models.
- **Model steps.**
  - At the edge view: BA(cloud fam, edge view) - BA(edge).
  - At the cloud view: BA(cloud) - BA(edge fam, cloud view).
- **Information steps.**
  - Under the edge family: BA(edge fam, cloud view) - BA(edge).
  - Under the cloud family: BA(cloud) - BA(cloud fam, edge view).
- **Shapley steps.** The mean of the two model steps and the mean of the two information steps.
- **Interaction.** BA(cloud) - BA(cf, ev) - BA(ef, cv) + BA(edge).
- **Stacker.** Two candidates, both fitted on validation: a logistic stacker on the logits of
  both models, and a gradient-boosting stacker (max_iter = 100, max_depth = 3) on both
  probabilities. Pick the one with the higher 3-fold grouped cross-validated BA inside the
  validation set, refit it on the whole validation set, and threshold at 0.5.
- **Combination step.** BA(stacker) - BA(cloud).
- **Unresolved disagreement.** BA(oracle) - BA(stacker).
- **Both edges.** Combination and unresolved disagreement are computed for the deployed edge
  and again for the upgraded edge.
- **Self-oracle headroom (pair A).** BA(oracle of cloud and its re-seeded twin) - max of the
  two BAs.

## 6. Routers

Each router produces a score on test windows; the highest scores are escalated first. Budgets
are f = 0.05, 0.10, 0.20, 0.30 and 0.50. The operating range for capture is 0.05 to 0.30.
p~ is the edge probability mapped so that its threshold sits at 0.5.

- **R0 random.** Seeded.
- **R1 confidence.** -|p~ - 0.5|.
- **R2 entropy.** Binary entropy of the raw edge probability.
- **R3 conformal.** Split conformal on validation with nonconformity 1 - p~(y). The score is
  min(p-value for 0, p-value for 1).
- **R4 Diff-01.** A small gradient-boosting regressor, trained on validation, predicts
  1[cloud right] - 1[edge right] from edge-view features and p_E.
- **R5 Diff-Prob.** The same regressor predicts p_cloud(y) - p_edge(y).
- **R6 metric-weighted gain.** The same regressor predicts g, using balanced-accuracy sample
  weights from the validation labels.
- **R7 = R6 plus explanation, stability and drift inputs (pair A only).** The added inputs
  are:
  - the top-8 sorted |attribution| normalised by L1;
  - log L1;
  - the attribution entropy;
  - the signed sum and the max |attribution|, where attributions are coefficient times
    standardised feature;
  - stability, the mean cosine similarity of attributions under five Gaussian perturbations
    with sd 0.05 in standardised units;
  - drift, the mean of |robust z| clipped at 10 against the fit-set median and IQR.
- **R8 learning to defer (pair A only).** The Mozannar-Sontag cross-entropy surrogate. A
  linear model on standardised edge features has three logits (negative, positive, defer).
  The loss is -log p(y) - 1[cloud right] log p(defer), with class-balanced weights on the y
  term. Cloud correctness on the fit set comes from 3-fold cross-fitting. The score is the
  defer logit minus the larger class logit.
- **R9 periodic.** Score = -frac(order x 0.618034).
- **Oracle ranking.** By g, using test labels. This is the ceiling.
- **Per-router outputs.** C, capture over [0.05, 0.30], and BA and recall at every budget.

## 7. Hypotheses and primary tests

Per-dataset effects use the Nadeau-Bengio corrected t-test. Pooled effects across the four
datasets use a random-effects model with the Hartung-Knapp-Sidik-Jonkman correction (t with
3 df). The five primary p-values are Holm-adjusted together.

- **H1. The information step is positive.**
  - Test: the pooled Shapley information step (pair A), one-sided.
  - Supported if the pooled p < 0.05 after Holm and at least two datasets are individually
    positive (one-sided p < 0.05).
  - Falsified if the pooled 95% CI includes 0 and no dataset is individually positive.
- **H2. Identifiability rises with labelled fault units.**
  - Design: in the training split only, keep positive windows in a random fraction
    q = 0.1, 0.25, 0.5 or 1.0 of the groups that contain positives (at least one group), and
    drop positives elsewhere. Negatives are untouched. Pair A, router R6, test C.
  - Test: the slope of C on log2(number of positive training windows) in a linear mixed model
    with dataset random intercepts. One-sided, slope > 0.
  - Falsified if the slope CI includes 0, or the mean C at q = 0.1 exceeds 0.6.
- **H3. Explanation and drift inputs do not raise C.**
  - Test: dC = C(R7) - C(R6) (pair A), pooled TOST with margin ±0.02.
  - Supported if p_eq < 0.05.
  - Falsified if at least three datasets show dC > 0 with one-sided p < 0.05 (Holm within H3).
  - If the corrected-SE floor makes the TOST infeasible, that is reported and no equivalence is
    claimed.
- **H4. Learned routers beat confidence only where confidence fails to order gains.**
  - For each dataset, pair (A, B) and split, let rho_pos be the Spearman correlation between
    the R1 score and g within the positives, and dcap = capture(R6) - capture(R1).
  - Test: dcap ~ rho_pos + (1 | dataset). One-sided, slope < 0.
  - Falsified if the slope is >= 0 or its CI includes 0.
- **H5. A pre-specified audit predicts whether routing pays.**
  - Audit: in each cell (dataset × pair A or B × split × H2 level), cross-fit R6 by 5-fold
    grouped CV inside the validation set and compute the validation C with a 1,000-resample
    group bootstrap CI. Call GO when the lower bound exceeds 0.52.
  - Realised outcome on test: capture(R6, 0.05-0.30) > 0.10.
  - Test: the balanced accuracy of the calls against the outcomes, with a dataset-stratified
    permutation p-value (10,000 permutations).
  - Supported if p < 0.05 and the lower bound of a dataset×split block-bootstrap 95% CI for the
    audit's balanced accuracy exceeds 0.5.

## 8. Secondary and descriptive analyses

- Per-dataset, per-pair tables of every quantity in Sections 5 and 6, with corrected 95% CIs.
- Payload in bytes per escalated window: edge raw, cloud raw, edge features, cloud features and
  the 96-B explanation token.
- Model footprint per family: parameter count, serialised size and multiply-accumulates per
  window. These are compared against published memory budgets of microcontroller- and
  gateway-class boards. The comparison is descriptive only, with no on-board timing claimed.
- Sensitivity analyses: composite thresholds; 3W grouped by well instead of instance; the
  router comparison at every budget.
- Pilots: the SECOM and MetroPT-3 results, labelled as pilots.

## 9. Power and feasibility

Corrected per-dataset standard errors cannot fall below sd·sqrt(n_test/n_train). For every
±0.02 equivalence test, report this floor and whether the TOST is achievable. The pooled
random-effects tests rest on four datasets, so their intervals are wide by design.

## 10. Reproducibility

- Code: `src/escalation/`.
- Per-dataset entry point: `src/run_dataset.py <name> <pair>`.
- Every split writes long-format rows to `results/<name>/<pair>/`.
- Seeds are fixed as above.
- The environment is recorded in `results/ENVIRONMENT.txt`.
- Every number in the paper is regenerated by `src/make_tables.py` from those CSVs.

## Addendum A: edge nodes (filled in from the dataset cards before each dataset's first run)

| Dataset | Edge node | Edge sensors | Cloud sensors | Groups | Recorded |
|---|---|---|---|---|---|
| hydraulic | | | | | |
| skab | | | | | |
| seu | | | | | |
| threew | | | | | |
