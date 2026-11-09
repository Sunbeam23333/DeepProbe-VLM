# Accepted experimental results

This snapshot publishes accepted R1/R2 seed17 evaluations and their paired statistics. Machine-readable figures and archive identifiers are in [accepted_results.json](accepted_results.json); original evidence digests are in [RESULTS_SOURCE_MAP.json](RESULTS_SOURCE_MAP.json). The JSON is a curated projection, not a copy of the private operational evidence.

## Video-MME: declared 899-video subset

Both models were evaluated on the same 2,697 questions from 899 videos. Video 490 and all three associated questions were excluded. This is not the official full 900-video result. The frozen protocol uses F8/G4, greedy decoding with 16 new tokens, no subtitles and no audio, and a strict reference multiple-choice scorer.

- R1: **6.3404%** accuracy (171/2,697), with 2,215 invalid responses (**82.1283%**).
- R2: **18.4650%** accuracy (498/2,697), with 1,409 invalid responses (**52.2432%**).
- Paired R2 − R1: **+12.1246 percentage points**, 95% CI **[10.6044, 13.5716]**.

Invalid responses count as wrong and remain in the denominator. The invalid-response rates substantially limit the absolute results; the reported improvement must not be described as strong general video understanding.

## TempCompass: accepted multiple-choice set

Both models were evaluated on all 1,580 accepted questions from 410 clips. The source map contains 188 accepted source-association groups covering 207 underlying sources. The frozen protocol uses F8/G4, greedy decoding with 16 new tokens, no subtitles and no audio, and the official `disable_llm` multiple-choice rules. Original official source parity was independently checked.

- R1: **42.5316%** accuracy (672/1,580); 1,421 responses matched the official rules.
- R2: **50.4430%** accuracy (797/1,580); 1,575 responses matched the official rules.
- Paired R2 − R1: **+7.9114 percentage points**, 95% CI **[5.1876, 10.6785]**.

Unmatched examples remain in the accepted data. TempCompass and Video-MME have different scoring protocols; their accuracies must not be pooled into one score.

## Statistical scope

The intervals use a paired source-cluster bootstrap with 2,000 resamples and random seed 17: 899 video clusters for Video-MME and 188 preaccepted source-association groups for TempCompass. The original predictions, correctness decisions, source identities and question order were retained. Overall estimates and intervals were independently recomputed; subgroup intervals were not included in that independent recomputation.

These intervals describe uncertainty from sampling test sources. They do not measure variability across training seeds: the accepted model comparison uses training seed17 only. Evaluation retries are not additional training seeds. The statistics do not establish robustness across model pretraining differences.

## Training and archive acceptance

R4 seed17 alignment completed 718 optimizer steps and exited successfully. Independent acceptance binds 143 archived objects / 15,934,454,991 bytes, including a final checkpoint of 20 files / 7,482,814,704 bytes. This is training and archive engineering acceptance; it is not a R4 held-out score. Current R4 SFT status is tracked separately from this static results snapshot.

## Evidence availability

The accepted evaluation archives contain 75 objects / 12,961,861 bytes for Video-MME and 119 objects / 11,353,379 bytes for TempCompass. Their original acceptance records verify every archived object through full download and SHA256 comparison. Exact OSS archive identifiers and manifest digests are included in the JSON. The buckets are access-controlled; no public artifact access is implied, and this curation did not repeat the remote download checks.

The repository summary excludes credentials, media, model weights, internal cluster manifests and deployment packages. Raw operational evidence is retained privately. The evidence index identifies its original bytes by SHA256; the public summary has a separate digest. Synthetic development-format diagnostics are not benchmark accuracy, and available timing records are not a warmed latency benchmark.
