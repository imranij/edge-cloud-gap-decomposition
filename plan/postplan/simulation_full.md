# Post-plan: full validation of the proposed test designs

Post-plan, not pre-specified. Nothing here changes a confirmatory verdict. Written and hashed
(plan/postplan/HASHES.txt) BEFORE the full run started; the pilot (results_sens/simulation/,
results_sens/simulation_r200/) is untouched and stays the cited pilot.

## Specification
- J = 20 splits per synthetic dataset (the plan's 4 repeats of K = 5), the designs as on the data:
  H2 fixed tiers, two-stage; H3 TOST with the corrected feasibility rule at the realised J; H4 Cap(R6) - Cap(R0)
  on validation rho_pos, two-stage with (i) the published factor sqrt(1 + J n_test/n_train), (ii) no factor,
  (iii) per-repeat slopes, naive SE over repeats, (iv) per-repeat slopes with an NB-type factor
  sqrt(1/R + n_test/n_train); the audit with the within-class null, B = 10 permutations in every split, GO
  threshold = 95th percentile of the 200 null values of the same dataset (as src/sens/designs/p1w_within_class.py),
  undefined cells reported separately; the original (plan) designs alongside.
- Replicates FIXED before the run (src/sens/simulation_full/worker.py, R_FIXED): S0 1000, S1 1000, S1b 500, S2 500,
  S3 500; T0_0.0, T0_0.3, T0_0.6, T0_0.9, T0s_0.9 1000 each; T1_0.0, T1_0.3, T1_0.6, T1_0.9 500 each; C_clock 1000,
  C_offset 1000, C_ref 500. Truth: 200 Monte Carlo draws per scenario. 13,000 replicate units + 340 truth chunks,
  about 97 CPU-hours.
- Seeds: dataset k of replicate r of scenario i: SeedSequence([20261013, i, r, k]) (the pilot used 20261012).

## Scenarios
- Base S0-S3: the pilot generator unchanged (src/sens/simulation/generator.py), full pipeline.
- Temporal dependence T: the hydraulic rig's own label sequence and 111-cycle blocks (20 blocks, 16 mixed),
  AR(1) block effects with lag-one correlation phi = 0, 0.3, 0.6, 0.9 on both views; T0 no information (size,
  share of splits with a positive step), T1 mu_c = 1 (coverage with a real step), T0s_0.9 tau = 1 (stress).
  Reported: coverage of the grouped NB interval for (A) the step on an independent series ("deployment truth")
  and (B) the cross-validated expectation; size; share of positive split steps; the cycle-index shortcuts and the
  survival criterion of Sec. 5.3.
- Confounded C: SKAB-like files with an injection schedule (12 anomaly-free and
  28 experiment files, onset U(0.1, 0.9)); no fault information in the cloud-only view, but a label-correlated
  clock (C_clock, kappa = 10) or file-type offset (C_offset, delta = 8) that reaches only the cloud view; C_ref has
  a real step (mu_c = 1) and no confound. Controls: within-file z-scoring (declared variant) and the shortcut
  variables added to both views; shortcut baselines time-in-file tree, file-type logistic, both tree.

## Calibration before hashing (2026-10-12, 20:10-20:20 UTC)
- C strengths: 24 draws of one split per setting (the calibration script was not kept). With 8
  anomaly-free files and onset U(0.3, 0.7) the metadata predicted the labels almost perfectly (tree on t and file
  type, BA 0.92), so the onset range was widened to U(0.1, 0.9) and 12 anomaly-free files used. The spurious plain
  step rose with kappa and delta and saturated near +0.03 (null scenario -0.014): kappa 3/5/6/10/14 gave
  +0.017/+0.026/+0.018/+0.024/+0.030 (different file counts), delta 3/5/8 gave +0.011/+0.037/+0.029. Chosen: kappa 10,
  delta 8, near saturation, so that each confound fakes a step of the size of the small real step (S1, S2).
- T and C_ref use mu_c = 1.0 (the S1b value) without tuning.

## Verification before the run
- The full base pipeline reproduces the pilot pipeline (src/sens/simulation/pipeline.run_sim_dataset, 10 splits)
  to 1.1e-16 on all 54 shared columns of S3 replicate 0, dataset 1; the within-class null draw 0 is identical.
- Timing (this machine, 1 CPU per unit): base 89 s, T 2.4 s, C 5.5 s per replicate; truth chunks of 10 draws
  16 s (base), 1.6 s (T), 3.7 s (C).
- Reference units (replicate 0 and truth chunk 0 of S0, S3, T1_0.9 and C_offset) are stored in
  results_sens/simulation_full/refs/. Every machine that took part in the run had to reproduce them with
  src/sens/simulation_full/verify_ref.py (max |diff| <= 0.05, reported) before it started. A fresh Python 3.13
  environment with the same package versions and without PyTorch (src/sens/simulation_full/torch_stub.py)
  reproduced all eight files exactly (max |diff| 0).
- summarize_full.py was unit-tested on hand-computed H4 examples. It may be corrected after the run without
  affecting the hashed generator, pipeline and R.

## Log (UTC)
- 2026-10-12T20:45Z  hashes written; the launcher for a computing cluster was packed with the code.
- 2026-10-12T20:55Z  launcher tested end to end against local stand-ins for SSH and Slurm (environment set-up with
  the same package versions and no PyTorch, verification identical, job array, summary, retrieval); the launcher
  was hashed again.
- 2026-10-12T21:05Z  the run was moved off the cluster. A Linux server (2 CPUs) ran the T and C families and then
  base slice 0 of 2 (worker --task 0 --ntasks 2). A Linux virtual machine on a desktop PC (2 CPUs, same package
  versions in a uv environment, no PyTorch) ran base slice 1 in reverse order, and the same PC under Windows later
  joined slice 1 forward and then slice 0. Units are deterministic and skipped if already present, so duplicates
  are harmless and the union is summarised. On the virtual machine the four reference replicates were identical
  (max |diff| 0) and the base truth chunks differed by at most 4.2e-7 (logistic regression on a 50,000-window test
  set, different CPU), reported as is.
- 2026-10-13T04:20Z  the server's base slice ran 80 minutes without finishing a unit. With the real PyTorch imported
  in the parent before forking, the forked gradient-boosting fits spun (the virtual machine, which has no PyTorch,
  ran normally). The pre-fork import is now done only with the stand-in (execution-only change to worker.py;
  results unaffected).
- 2026-10-13T04:24Z  the server slice was restarted in a uv environment with the same package versions and no
  PyTorch, at 83 s per base unit. With the system Python (PyTorch installed) two concurrent workers had made each
  gradient-boosting fit take about 3 s instead of 0.03 s.
- 2026-10-13T10:00Z  all 13,000 replicate units and 340 truth chunks finished. The Windows PC (19 worker processes,
  same package versions in a uv environment) ran most of the base family from 07:16 to 09:56 UTC; the cluster
  verification job had passed but the job array stayed pending in the queue and was cancelled. Canonical set
  results_sens/simulation_full/{reps,truth} (provenance.csv) has the T and C families from the server, and base units
  from the server where it computed them and otherwise from the desktop PC (virtual machine or Windows).
  Cross-platform check on the 436 base units computed both on the server (Linux) and on the PC: 422 agree to 1e-3
  (398 exactly); in 14 a split's validation threshold fell on a tie that the two platforms broke differently
  (floating-point differences in the logistic-regression probabilities), changing that split's downstream values.
  Both versions are valid draws of the same design, and the server version is kept where both exist.
- 2026-10-13T10:12Z  three base files copied from the virtual machine were truncated by an interrupted copy (S1b r83,
  S2 r197, S3 r457); the units were recomputed on the server (provenance: server_recomputed). Every replicate file now
  has 80 rows and every truth chunk 10 draws.
- 2026-10-13T10:40Z  summarised (results_sens/simulation_full/summary/). The supplement tables of the full validation,
  the column "Proposed corrected design" of the test table and the related sentences of the paper were written from
  these files, and every new claim was checked against them.
