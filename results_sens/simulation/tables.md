Replicates per scenario: S0 105, S1 105, S1b 104, S2 104, S3 104

#### Ground truth (Monte Carlo mean over draws, MC SE in parentheses)

| quantity | S0 | S1 | S1b | S2 | S3 |
|---|---|---|---|---|---|
| info_step | -0.0063 (0.0012) | 0.0305 (0.0017) | 0.0799 (0.0014) | 0.0400 (0.0015) | 0.0800 (0.0011) |
| info_step_fit800 | -0.0066 (0.0012) | 0.0294 (0.0013) | 0.0814 (0.0014) | 0.0397 (0.0011) | 0.0796 (0.0011) |
| ba_edge | 0.7973 (0.0014) | 0.7986 (0.0012) | 0.7979 (0.0012) | 0.7972 (0.0014) | 0.7989 (0.0011) |
| ba_cloud | 0.7910 (0.0012) | 0.8292 (0.0015) | 0.8779 (0.0011) | 0.8373 (0.0009) | 0.8789 (0.0008) |
| share_g_nonzero | 0.0767 (0.0029) | 0.1335 (0.0030) | 0.1663 (0.0020) | 0.1453 (0.0028) | 0.1612 (0.0015) |
| C_R0 | 0.5004 (0.0005) | 0.4997 (0.0004) | 0.5005 (0.0003) | 0.4995 (0.0004) | 0.4999 (0.0004) |
| C_R1 | 0.4785 (0.0049) | 0.5999 (0.0033) | 0.6631 (0.0025) | 0.6097 (0.0028) | 0.6373 (0.0020) |
| C_R6 | 0.5065 (0.0028) | 0.5297 (0.0034) | 0.5671 (0.0038) | 0.6026 (0.0043) | 0.6565 (0.0040) |
| dC_R6_R1 | 0.0280 (0.0059) | -0.0702 (0.0035) | -0.0960 (0.0033) | -0.0071 (0.0053) | 0.0192 (0.0045) |
| cap_R1 | -0.0278 (0.0142) | 0.2236 (0.0051) | 0.2787 (0.0034) | 0.2313 (0.0046) | 0.2208 (0.0029) |
| cap_R6 | -0.0036 (0.0065) | 0.0811 (0.0067) | 0.1245 (0.0059) | 0.2054 (0.0087) | 0.2653 (0.0082) |
| dcap_R6_R1 | 0.0242 (0.0126) | -0.1425 (0.0057) | -0.1542 (0.0054) | -0.0259 (0.0100) | 0.0445 (0.0086) |
| ba10_R1_minus_edge | 0.0013 (0.0006) | 0.0159 (0.0005) | 0.0275 (0.0004) | 0.0183 (0.0006) | 0.0236 (0.0003) |
| ba10_R6_minus_edge | -0.0005 (0.0004) | 0.0081 (0.0006) | 0.0173 (0.0005) | 0.0177 (0.0007) | 0.0276 (0.0007) |
| C_R6_gt_half | 0.5900 (0.0494) | 0.7800 (0.0416) | 0.9700 (0.0171) | 0.9900 (0.0100) | 1.0000 (0.0000) |
| dC_R6a_R6b | 0.0002 (0.0008) | -0.0008 (0.0009) | 0.0004 (0.0009) | 0.0003 (0.0011) | -0.0010 (0.0012) |
| rho_pos_test | -0.0316 (0.0263) | 0.1169 (0.0201) | 0.2039 (0.0138) | 0.1182 (0.0202) | 0.1863 (0.0127) |
| h2f_slope_C_R6 | 0.0002 (0.0018) | 0.0087 (0.0020) | 0.0162 (0.0024) | 0.0233 (0.0026) | 0.0358 (0.0026) |
| h2j_slope_C_R6 | -0.0062 (0.0015) | 0.0041 (0.0018) | 0.0118 (0.0019) | 0.0183 (0.0024) | 0.0205 (0.0027) |
| h2j_slope_dC_R6_R1 | -0.0007 (0.0026) | -0.0099 (0.0024) | -0.0043 (0.0022) | 0.0049 (0.0030) | 0.0128 (0.0030) |

Draws per scenario: S0 100, S1 100, S1b 100, S2 100, S3 100

#### H1: information-step tests

| design | quantity | S0 | S1 | S1b | S2 | S3 |
|---|---|---|---|---|---|---|
| per-dataset NB one-sided | rejection rate (step > 0) | 0.000 (0.000) | 0.345 (0.023) | 0.950 (0.010) | 0.430 (0.025) | 0.969 (0.008) |
| pooled HKSJ one-sided | rejection rate (step > 0) | 0.000 (0.000) | 0.714 (0.044) | 1.000 (0.000) | 0.846 (0.036) | 1.000 (0.000) |
| plan verdict rule | SUPPORTED rate | 0.000 (0.000) | 0.381 (0.048) | 1.000 (0.000) | 0.587 (0.049) | 1.000 (0.000) |
| per-dataset NB 95% CI | coverage of true step (fit 600) | 0.990 (0.005) | 0.986 (0.006) | 0.971 (0.009) | 0.986 (0.006) | 0.976 (0.007) |
| per-dataset NB 95% CI | coverage of true step (fit 800) | 0.990 (0.005) | 0.983 (0.006) | 0.971 (0.009) | 0.986 (0.006) | 0.976 (0.007) |
| per-dataset naive t 95% CI | coverage of true step (fit 600) | 0.921 (0.012) | 0.848 (0.019) | 0.827 (0.020) | 0.825 (0.019) | 0.853 (0.016) |
| pooled HKSJ 95% CI | coverage of true step (fit 600) | 0.990 (0.010) | 0.990 (0.010) | 0.990 (0.010) | 1.000 (0.000) | 1.000 (0.000) |
| per-dataset NB one-sided | size at truth, upper tail | 0.000 (0.000) | 0.026 (0.010) | 0.022 (0.008) | 0.010 (0.005) | 0.007 (0.004) |
| per-dataset NB one-sided | size at truth, lower tail | 0.019 (0.007) | 0.024 (0.007) | 0.024 (0.007) | 0.031 (0.008) | 0.036 (0.009) |
| pooled HKSJ one-sided | size at truth, upper tail | 0.000 (0.000) | 0.010 (0.010) | 0.010 (0.010) | 0.000 (0.000) | 0.000 (0.000) |
| pooled HKSJ one-sided | size at truth, lower tail | 0.019 (0.013) | 0.010 (0.010) | 0.000 (0.000) | 0.038 (0.019) | 0.000 (0.000) |

#### H3: TOST at +-0.02 (dC = C(R6a) - C(R6b))

| design | quantity | S0 | S1 | S1b | S2 | S3 |
|---|---|---|---|---|---|---|
| pooled TOST +-0.02 (true dC = 0) | equivalence declared | 0.0095 (0.0095) | 0.0476 (0.0209) | 0.0769 (0.0263) | 0.0385 (0.0189) | 0.1635 (0.0364) |
| original feasibility rule (asymptotic floor) | predicted feasible | 0.0762 (0.0260) | 0.2762 (0.0438) | 0.6442 (0.0472) | 0.5000 (0.0493) | 0.8365 (0.0364) |
| corrected feasibility rule (realised J, HKSJ) | predicted feasible | 0.0095 (0.0095) | 0.1048 (0.0300) | 0.2885 (0.0446) | 0.1827 (0.0381) | 0.5481 (0.0490) |
| pooled TOST | declared AND original rule feasible | 0.0095 (0.0095) | 0.0476 (0.0209) | 0.0769 (0.0263) | 0.0385 (0.0189) | 0.1635 (0.0364) |
| pooled TOST | declared AND corrected rule feasible | 0.0095 (0.0095) | 0.0476 (0.0209) | 0.0769 (0.0263) | 0.0385 (0.0189) | 0.1635 (0.0364) |
| pooled TOST +-0.02 (true dC = 0) | equivalence declared | original rule feasible | 0.1250 (0.1250) | 0.1724 (0.0714) | 0.1194 (0.0399) | 0.0769 (0.0373) | 0.1954 (0.0428) |
| pooled TOST +-0.02 (true dC = 0) | equivalence declared | corrected rule feasible | 1.0000 | 0.4545 (0.1575) | 0.2667 (0.0821) | 0.2105 (0.0961) | 0.2982 (0.0611) |
| feasibility rules | original feasible, corrected not | 0.0667 (0.0245) | 0.1714 (0.0370) | 0.3558 (0.0472) | 0.3173 (0.0459) | 0.2885 (0.0446) |
| per-dataset TOST +-0.02 (true dC = 0) | equivalence declared | 0.0024 (0.0024) | 0.0095 (0.0047) | 0.0072 (0.0041) | 0.0000 (0.0000) | 0.0216 (0.0069) |
| per-dataset original rule | predicted feasible | 0.0071 (0.0041) | 0.0286 (0.0078) | 0.0889 (0.0152) | 0.0625 (0.0131) | 0.1130 (0.0152) |
| per-dataset corrected rule | predicted feasible | 0.0024 (0.0024) | 0.0190 (0.0065) | 0.0337 (0.0091) | 0.0144 (0.0075) | 0.0601 (0.0111) |
| pooled TOST, true dC = +0.02 | equivalence declared (size) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| per-dataset TOST, true dC = +0.02 | equivalence declared (size) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| pooled HKSJ 95% CI | coverage of true dC = 0 | 1.0000 (0.0000) | 1.0000 (0.0000) | 1.0000 (0.0000) | 1.0000 (0.0000) | 1.0000 (0.0000) |
| h3_sd_dC | mean over replicates | 0.0552 (0.0012) | 0.0416 (0.0007) | 0.0356 (0.0006) | 0.0378 (0.0005) | 0.0312 (0.0005) |
| h3_pool_se | mean over replicates | 0.0145 (0.0003) | 0.0112 (0.0002) | 0.0097 (0.0002) | 0.0100 (0.0002) | 0.0085 (0.0001) |
| h3_pool_floor | mean over replicates | 0.0121 (0.0003) | 0.0094 (0.0002) | 0.0081 (0.0001) | 0.0085 (0.0001) | 0.0071 (0.0001) |

#### H4

| design | quantity | S0 | S1 | S1b | S2 | S3 |
|---|---|---|---|---|---|---|
| original (test rho, R6 - R1, MixedLM) | one-sided p < 0.05 | 0.438 (0.049) | 0.419 (0.048) | 0.365 (0.047) | 0.413 (0.049) | 0.567 (0.049) |
| original (test rho, R6 - R1, MixedLM) | SUPPORTED rate | 0.362 (0.047) | 0.343 (0.047) | 0.298 (0.045) | 0.317 (0.046) | 0.500 (0.049) |
| original with R0 in place of R6 | SUPPORTED rate | 0.371 (0.047) | 0.514 (0.049) | 0.452 (0.049) | 0.442 (0.049) | 0.510 (0.049) |
| validation rho only (R6 - R1) | SUPPORTED rate | 0.038 (0.019) | 0.019 (0.013) | 0.010 (0.010) | 0.019 (0.014) | 0.029 (0.016) |
| R0 reference only (test rho) | SUPPORTED rate | 0.010 (0.010) | 0.000 (0.000) | 0.038 (0.019) | 0.010 (0.010) | 0.058 (0.023) |
| corrected (val rho, R6 - R0, MixedLM) | one-sided p < 0.05 | 0.038 (0.019) | 0.038 (0.019) | 0.029 (0.016) | 0.010 (0.010) | 0.048 (0.021) |
| corrected (val rho, R6 - R0, MixedLM) | SUPPORTED rate | 0.038 (0.019) | 0.029 (0.016) | 0.000 (0.000) | 0.010 (0.010) | 0.038 (0.019) |
| corrected (val rho, R6 - R0, two-stage HKSJ) | one-sided p < 0.05 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| MixedLM fallback to OLS | share of replicates | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| h4_orig_slope | mean over replicates | -0.499 (0.032) | -0.229 (0.017) | -0.199 (0.017) | -0.259 (0.019) | -0.301 (0.018) |
| h4_origR0_slope | mean over replicates | -0.573 (0.030) | -0.307 (0.017) | -0.271 (0.015) | -0.294 (0.018) | -0.303 (0.015) |
| h4_cor_mixed_slope | mean over replicates | -0.041 (0.019) | 0.027 (0.015) | 0.093 (0.019) | 0.051 (0.018) | 0.052 (0.018) |
| h4_cor_2stage_slope | mean over replicates | -0.008 (0.019) | 0.027 (0.018) | 0.098 (0.022) | 0.056 (0.019) | 0.052 (0.020) |
| h4_rho_val_test_spearman | mean over replicates | 0.760 (0.007) | 0.590 (0.012) | 0.407 (0.014) | 0.557 (0.013) | 0.482 (0.012) |

#### H2

| design | quantity | S0 | S1 | S1b | S2 | S3 |
|---|---|---|---|---|---|---|
| original (joint retraining, MixedLM) | one-sided p < 0.05 | 0.019 (0.013) | 0.124 (0.032) | 0.529 (0.049) | 0.885 (0.031) | 1.000 (0.000) |
| original (joint retraining, MixedLM) | SUPPORTED rate | 0.019 (0.013) | 0.086 (0.027) | 0.404 (0.048) | 0.827 (0.037) | 0.808 (0.039) |
| joint retraining, two-stage HKSJ (diagnostic) | one-sided p < 0.05 | 0.000 (0.000) | 0.010 (0.010) | 0.067 (0.025) | 0.260 (0.043) | 0.673 (0.046) |
| corrected (fixed tiers, C(R6) - C(R1), two-stage) | one-sided p < 0.05 | 0.000 (0.000) | 0.000 (0.000) | 0.038 (0.019) | 0.154 (0.036) | 0.760 (0.042) |
| level C(R6) - C(R1) at q = 1, two-stage (supplementary) | one-sided p < 0.05 | 0.038 (0.019) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.067 (0.025) |
| h2_orig_slope | mean over replicates | -0.005 (0.001) | 0.002 (0.001) | 0.010 (0.001) | 0.018 (0.001) | 0.025 (0.001) |
| h2_orig_mean_C_q01 | mean over replicates | 0.523 (0.002) | 0.520 (0.001) | 0.536 (0.001) | 0.543 (0.001) | 0.589 (0.001) |
| h2_cor_slope | mean over replicates | 0.001 (0.001) | 0.006 (0.001) | 0.013 (0.001) | 0.020 (0.001) | 0.033 (0.001) |
| h2_level_dC | mean over replicates | 0.035 (0.004) | -0.075 (0.002) | -0.099 (0.001) | -0.013 (0.002) | 0.021 (0.002) |

#### H5: audit calls (cells = dataset x split; MC SE by replicate bootstrap)

| design | quantity | S0 | S1 | S1b | S2 | S3 |
|---|---|---|---|---|---|---|
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | go_rate | 0.062 (0.004) | 0.053 (0.003) | 0.069 (0.004) | 0.155 (0.006) | 0.281 (0.007) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | base rate of (i) test C(R6) > 0.5 | 0.515 (0.008) | 0.611 (0.008) | 0.790 (0.007) | 0.860 (0.006) | 0.977 (0.002) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | base rate of (ii) BA at 10% by R6 > edge | 0.278 (0.007) | 0.632 (0.008) | 0.850 (0.006) | 0.832 (0.007) | 0.969 (0.003) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | base rate of (iii) Cap(R6) > 0.10 | 0.307 (0.007) | 0.420 (0.007) | 0.556 (0.009) | 0.698 (0.008) | 0.892 (0.005) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | precision vs (i) test C(R6) > 0.5 | 0.546 (0.032) | 0.652 (0.029) | 0.833 (0.025) | 0.914 (0.010) | 0.985 (0.003) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | recall vs (i) test C(R6) > 0.5 | 0.066 (0.005) | 0.057 (0.004) | 0.073 (0.004) | 0.164 (0.007) | 0.284 (0.007) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | ba vs (i) test C(R6) > 0.5 | 0.504 (0.004) | 0.505 (0.003) | 0.509 (0.005) | 0.535 (0.006) | 0.552 (0.019) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | precision vs (ii) BA at 10% by R6 > edge | 0.309 (0.026) | 0.643 (0.034) | 0.830 (0.019) | 0.888 (0.013) | 0.977 (0.005) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | recall vs (ii) BA at 10% by R6 > edge | 0.069 (0.007) | 0.054 (0.005) | 0.068 (0.004) | 0.165 (0.007) | 0.283 (0.007) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | ba vs (ii) BA at 10% by R6 > edge | 0.505 (0.004) | 0.501 (0.004) | 0.494 (0.005) | 0.531 (0.007) | 0.536 (0.018) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | precision vs (iii) Cap(R6) > 0.10 | 0.347 (0.028) | 0.442 (0.034) | 0.611 (0.029) | 0.804 (0.014) | 0.927 (0.007) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | recall vs (iii) Cap(R6) > 0.10 | 0.071 (0.007) | 0.056 (0.005) | 0.076 (0.005) | 0.178 (0.007) | 0.292 (0.007) |
| original (GroupKFold, pooled OOF C, bootstrap LB > 0.52) | ba vs (iii) Cap(R6) > 0.10 | 0.506 (0.004) | 0.502 (0.003) | 0.508 (0.004) | 0.539 (0.005) | 0.552 (0.010) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | go_rate | 0.066 (0.004) | 0.075 (0.005) | 0.145 (0.006) | 0.168 (0.006) | 0.304 (0.008) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | precision vs (i) test C(R6) > 0.5 | 0.523 (0.032) | 0.682 (0.024) | 0.860 (0.014) | 0.903 (0.011) | 0.981 (0.004) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | recall vs (i) test C(R6) > 0.5 | 0.068 (0.006) | 0.083 (0.006) | 0.158 (0.007) | 0.176 (0.007) | 0.305 (0.008) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | ba vs (i) test C(R6) > 0.5 | 0.501 (0.004) | 0.511 (0.004) | 0.531 (0.006) | 0.530 (0.007) | 0.526 (0.023) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | precision vs (ii) BA at 10% by R6 > edge | 0.247 (0.024) | 0.678 (0.026) | 0.882 (0.013) | 0.877 (0.013) | 0.979 (0.004) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | recall vs (ii) BA at 10% by R6 > edge | 0.059 (0.006) | 0.080 (0.006) | 0.150 (0.007) | 0.177 (0.007) | 0.307 (0.008) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | ba vs (ii) BA at 10% by R6 > edge | 0.495 (0.004) | 0.507 (0.004) | 0.518 (0.007) | 0.527 (0.007) | 0.548 (0.019) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | precision vs (iii) Cap(R6) > 0.10 | 0.312 (0.028) | 0.490 (0.030) | 0.653 (0.019) | 0.784 (0.017) | 0.920 (0.008) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | recall vs (iii) Cap(R6) > 0.10 | 0.068 (0.008) | 0.087 (0.007) | 0.170 (0.009) | 0.189 (0.008) | 0.314 (0.008) |
| corrected (SGKF, fold-mean C > pooled within-class null 95th pct) | ba vs (iii) Cap(R6) > 0.10 | 0.501 (0.004) | 0.511 (0.004) | 0.529 (0.005) | 0.534 (0.006) | 0.544 (0.011) |
| corrected, D-weighted fold mean (variant) | go_rate | 0.067 (0.004) | 0.064 (0.004) | 0.127 (0.006) | 0.169 (0.006) | 0.307 (0.007) |
| corrected, D-weighted fold mean (variant) | precision vs (i) test C(R6) > 0.5 | 0.536 (0.032) | 0.670 (0.026) | 0.860 (0.016) | 0.902 (0.010) | 0.981 (0.004) |
| corrected, D-weighted fold mean (variant) | recall vs (i) test C(R6) > 0.5 | 0.069 (0.006) | 0.070 (0.005) | 0.139 (0.006) | 0.177 (0.006) | 0.309 (0.007) |
| corrected, D-weighted fold mean (variant) | ba vs (i) test C(R6) > 0.5 | 0.503 (0.004) | 0.508 (0.003) | 0.527 (0.006) | 0.529 (0.007) | 0.528 (0.023) |
| corrected, D-weighted fold mean (variant) | precision vs (ii) BA at 10% by R6 > edge | 0.271 (0.023) | 0.678 (0.030) | 0.889 (0.012) | 0.866 (0.015) | 0.980 (0.004) |
| corrected, D-weighted fold mean (variant) | recall vs (ii) BA at 10% by R6 > edge | 0.065 (0.007) | 0.068 (0.005) | 0.133 (0.006) | 0.176 (0.006) | 0.311 (0.007) |
| corrected, D-weighted fold mean (variant) | ba vs (ii) BA at 10% by R6 > edge | 0.499 (0.004) | 0.506 (0.004) | 0.519 (0.006) | 0.521 (0.007) | 0.558 (0.017) |
| corrected, D-weighted fold mean (variant) | precision vs (iii) Cap(R6) > 0.10 | 0.350 (0.028) | 0.498 (0.032) | 0.664 (0.022) | 0.778 (0.018) | 0.923 (0.007) |
| corrected, D-weighted fold mean (variant) | recall vs (iii) Cap(R6) > 0.10 | 0.076 (0.008) | 0.075 (0.006) | 0.152 (0.009) | 0.189 (0.007) | 0.318 (0.007) |
| corrected, D-weighted fold mean (variant) | ba vs (iii) Cap(R6) > 0.10 | 0.507 (0.004) | 0.510 (0.004) | 0.528 (0.006) | 0.532 (0.006) | 0.549 (0.010) |

#### H5: corrected-audit null calibration (pooled within-class permutation draws)

| scenario   |   n_null |   thr95_mean |   thr95_wmean |   null_mean |   null_sd |   obs_mean |   orig_C_val_mean |
|:-----------|---------:|-------------:|--------------:|------------:|----------:|-----------:|------------------:|
| S0         |      839 |       0.7371 |        0.7176 |      0.5318 |    0.1287 |     0.4962 |            0.4652 |
| S1         |      840 |       0.6727 |        0.6664 |      0.5204 |    0.0991 |     0.5069 |            0.477  |
| S1b        |      832 |       0.6467 |        0.6491 |      0.5125 |    0.0845 |     0.5427 |            0.5077 |
| S2         |      832 |       0.6718 |        0.6632 |      0.5209 |    0.0923 |     0.5653 |            0.5363 |
| S3         |      832 |       0.6686 |        0.6618 |      0.5208 |    0.0891 |     0.6163 |            0.5851 |
