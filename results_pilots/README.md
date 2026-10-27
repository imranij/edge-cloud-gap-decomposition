# Pilot results (not confirmatory)

These analyses on the SECOM and MetroPT-3 data sets were run on 11 October 2026, before the analysis plan was
frozen. They suggested the hypotheses H1 to H5 and are not used in any confirmatory test. The code that produced
them belongs to an earlier exploratory project and is not part of this repository. The files are kept so that the
pilot numbers quoted in the paper can be checked.

The main pilot results are listed below.
- SECOM, 20 overlapping resplits, Nadeau-Bengio corrected:
  - Edge-to-cloud gap +0.034 (CI -0.042 to +0.109).
  - Shapley model step +0.035 (-0.020 to +0.089).
  - Shapley information step -0.001 (-0.056 to +0.053).
  - Combination step +0.017.
  - Unresolved disagreement +0.128 (+0.059 to +0.196).
  - Confidence-ranking C 0.549 (0.466 to 0.633).
- SECOM edge view is label-selected (SelectKBest), not placement-defined.
- On MetroPT-3 both models flag none of the July failure windows (recall 0). The best stream drift
  detector reaches precision 0.0225 at recall >= 0.10.
