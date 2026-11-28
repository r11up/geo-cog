# Analysis plan and decision log of the revision

The analysis decisions of the revision, in the order they were made, with the date and local time
(UTC+05:45) at which each was recorded. "Before running" means the entry was written before the analysis it
governs was run; the other entries say what had been seen. Every variant that was run, including those not
in the paper, is logged with its configuration and result; the log is available from the authors. Each
analysis is recorded here before it is run.

## 25 November 2026

- **Before any model was trained.** EAV with the literature-informed prior (PECM) is re-run with the
  reviewers' controls (pure-noise environmental input; non-environmental structured input), and Urban
  Appraisal (OpenNeuro ds006850), in which the environment of each EEG epoch is the scene in view, is added
  as a co-registered extension. Pure-noise, shuffled-scene and low-level image controls for every scene-based
  model. The scale mismatch is treated as a change-of-support problem; if the dataset authors share the scene
  locations, image greenery is compared with Sentinel-2 NDVI within 100 m, 250 m and 1 km.
- **Before the full analysis.** Urban Appraisal participant sub-91 excluded (its EEG files end early; 222 of
  504 rated trials have no EEG), leaving 62. Two single missing ratings are imputed with the person's mean in
  training and skipped in scoring. Greenery-only and leave-one-category-out scene models added.
- **13:10, before any contrast-analysis result was seen.** The original EAV design is re-run in full with the
  original architecture (EEG tower over channel tokens of DE/PSD features; two-layer graph network over the
  grid; bidirectional cross-modal attention; squared-l2 alignment loss, lambda 0.1), plus the controls.
  Spatial scale tested at 1, 2 and 5 km cells. A simplified-network run of the same conditions is kept as a
  robustness check.
- **14:30, after the simplified run had been seen, before any full Study 1 result.** The Singapore
  domain-shift analysis is dropped: its OpenStreetMap build failed on the dense central tiles, and with the
  environmental input drawn independently of the label at test, another city's grid tests nothing the
  pure-noise, matched-random and removed-at-test controls do not.
- **22:45, amended 22:55, before running.** Strongest defensible EEG representation for every model: band-wise
  covariances (five bands, OAS shrinkage) re-centred at each participant's Riemannian mean without labels and
  mapped to the tangent space, with DE/PSD features; trial-level accuracy (mean of the four 5 s epoch
  probabilities) reported throughout; EEG-tower dropout and weight decay chosen per fold by validation
  macro-F1. Lockbox: EAV participants 1, 2, 5, 7, 9, 10, 12, 14, 27, 31, 36 and 40 (drawn with seed 20261125)
  are excluded from all exploration and scored once. Exploration on the other 30 with subject-level five-fold
  cross-validation; every variant logged; the EEG-only pipeline with the best development trial accuracy is
  selected, ties going to the simpler one.
- **23:20, after a literature check, before running.** Protocol P1 of the EAV paper (per participant, first
  70% of trials in time order for training, last 30% for testing, 5 s segments; last training trial of each
  class for validation). The group-level appraisal benchmark of Tan and Deng (group means of 55 scenes,
  11-fold cross-validation over scenes, ridge, mean Pearson r over nine scales). A planned time-resolved
  representational similarity analysis is dropped because Tan and Deng report one.

## 26 November 2026

- **00:10, after the development exploration, no lockbox data seen.** Selected EEG representation: DE/PSD with
  re-centred tangent-space covariances; best EEG-only model: logistic regression. The proposed model draws the
  environmental vector without the label in training as well as at test (label-free pairing); the
  label-conditioned PECM is kept as the ablation with all controls. Final evaluations run once: lockbox,
  subject-level cross-validation on all 42, and P1.
- **01:10, before running.** Image benchmark under the Tan and Deng protocol (20 random partitions; ridge
  penalty by leave-one-out; split-half noise ceiling); one pre-specified ensemble (mean of the out-of-fold
  predictions of all feature sets). Person-level prediction for people and scenes held out together. Decoding
  the viewed environment from EEG (ERP windows, band covariances; AUC per held-out person) with shuffled-label,
  low-level and residualised-greenery controls.
- **02:10, before running.** Protocols of published EAV studies: P2 (Gong et al.: random trials 60/10/30 per
  participant, 20 s trials, five assignments) and P3 (Lee, Suresh Chandra and Hansen: leave one subject out,
  2 s epochs, balanced accuracy, four training participants for validation). No tuning against any test set.
- **02:50, before running.** Time-resolved decoding (50 ms windows, sign-flip cluster permutation, 5,000
  permutations); EEGNet and EEG-Conformer on single trials; frozen CBraMod features with a linear probe.
- **03:05, before running.** P4: random 70/30 split of each participant's 5 s segments, as a reference only
  for published results whose split is not stated.
- **Entry time recorded as 01:35, written after the 03:05 entry, before running.** Protocol-leakage analysis for
  decoding greenery: people and scenes held out together, people only, scenes only, random person-scene pairs.
- **04:40, correction before re-running.** In one cell of the single-trial decoding, the six validation scenes
  were all of one class, so early stopping could not run; validation scenes are now drawn three per class. The
  earlier run is kept in the log.
- **09:13, before running (round 4).** Further EEG methods under P1, P2 and P3, same splits and metrics:
  drift correction (P1), seven-band filter bank, training windows every 1 s, soft vote with shrinkage LDA, all
  together. The reported method per protocol is the candidate with the best validation score, never the test
  score (P1: first 50 training trials against the next 20; P2: validation trials; P3: validation
  participants); every candidate is reported.
- **09:35, before running (round 4b).** EEG foundation model CBraMod: fine-tuned for P1 (alone and averaged
  with logistic regression) and as frozen features for P1 and P3.
- **11:38, amended 11:39, before running (round 5).** P1: CBraMod with a 50-epoch cosine schedule (alone and
  averaged with logistic regression), per-participant logistic regression, pooled and per-participant mean,
  transductive self-training. P3: self-training on the held-out participant, training participants weighted by
  EEG similarity. Selection across all rounds by validation.
- **11:49, before running.** Image round v4: four further encoders and ten-view test-time averaging;
  pre-specified equal-weight ensemble.
- **11:52, before running.** P5, the protocol of Malallah and Iqbal (2025): per participant, half of the 5 s
  segments for testing, assumed random within class.
- **11:59, before running.** CBraMod fine-tuned to decode high versus low greenery from single trials, same
  design as EEGNet and EEG-Conformer.
- **12:00, before computing.** Equivalence tests for the two Study 1 primary comparisons (proposed model vs
  EEG tower; PECM vs matched random): two one-sided paired t-tests, margin +/-3 percentage points.
- **12:42, before running.** Image round v5: final embedding with the mean of the last-layer patch tokens
  (primary per encoder) and three layers fixed in advance (one half, three quarters, full depth); no layer
  chosen on results; pre-specified equal-weight ensemble.
- **13:20, before computing.** Holm adjustment over the nine primary correlations of the extension
  (composition, EEG and fusion for valence, arousal and stress; two-sided Wilcoxon signed-rank tests).
- **14:38, results recorded.** In the selection split of round 5, per-participant models chose their penalty
  on the same validation windows, which favours them slightly in selection; every candidate's test result is
  reported, and no conclusion depends on the choice.
- **15:12, before computing.** Descriptive panel for the scene-composition figure: Spearman correlations over
  the 56 scenes between the five composition shares and the scene means of the nine rating scales (62
  participants). Descriptive only; no tests and nothing selected on it.
- **21:57, before running (round 6).** Study 1 fusion-operator controls on the same cross-validation folds,
  seeds and per-fold hyperparameters: single-direction cross-attention with a residual path, concatenation
  fusion, token-level cross-attention between the 31 EEG tokens and the 100 grid-node tokens in both directions
  and in one direction, the token-level model with pure-noise input, and the PECM prior and its matched-random
  control under concatenation and token-level fusion. Random Forest, LSTM and environment-only baselines re-run
  on the same folds and features with trial accuracy. Parameter counts reported. Holm adjustment over the
  paired comparisons of the ablation table. Urban Appraisal: the prior's directional assumptions tested on
  the co-registered ratings (greenery share and a built exposure score against valence and stress; per-person
  Spearman, Wilcoxon, Holm over four tests; the prior's quartile rule applied to the 56 scenes as a secondary
  check); a variance partition of appraisal prediction into composition, low-level statistics and shared
  parts; greenery residualised on low-level statistics; greenery share against the HT/LT design code. Images:
  grayscale and phase-scrambled controls for four encoders under the Tan and Deng protocol; two further
  segmentation models (Mask2Former trained on ADE20K and on Cityscapes) and a SegFormer-B5 re-run, compared
  with the released segmentation. Everything reported whatever the result.
- **22:15, before running (round 6 amendment).** Scene-level bootstrap of the image benchmark: out-of-fold
  predictions recomputed from the saved embeddings of the first round and of round v5, then 2,000 resamples of
  the 55 scenes with the predictions held fixed; intervals for the reproduction sets, the v5 readout of the
  same encoders, both pre-specified ensembles and the composition sets, and paired differences against the
  reproduction of the same encoder. 22:21: extended to the round-3 and round-4 pre-specified ensembles.
- **23:51, results recorded (round 6).** The Holm family declared for the ablation table spans that table and the
  separate fusion-operator table (14 paired comparisons, adjusted as one family). Split-wise paired differences
  are computed from the saved predictions of the cross-validation runs.

## 28 November 2026

- **22:21.** The freeze date of 1 December 2026 given in earlier versions of this file is withdrawn.
  Analyses may be added until submission; each is recorded here before it is run, as before.
