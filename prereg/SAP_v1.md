# Motion Canonical round 2: Statistical Analysis Plan

Version 1, fixed 29 September 2026 · Avnish Deobhakta, MD

## Status and scope

All data are collected and nothing has been scored. This plan is fixed before any response is compared with ground truth; it is committed to the project's GitHub repository, with a hash manifest of all collected data, before the analysis notebook joins `truth.csv`.

| Run | Contents | Calls | Collected (UTC) |
| --- | --- | --- | --- |
| `mc2_20260928T052340Z` | Primary: 4 arms x 4 vendors, vendor-default settings | 12,460 | 28 Sep 23:47 to 29 Sep 02:33 |
| `mc2reason_20260929T040656Z` | Sensitivity: ladder arm, Claude adaptive thinking at effort high, Astra effort high | 1,500 | 29 Sep 04:07 to 04:43 |
| `mc2maxreason_20260929T045331Z` | Sensitivity: ladder arm, Claude effort max, Astra effort xhigh | 1,500 | 29 Sep 04:55 to 06:12 |

Blinding rule: responses stay joined to image IDs only. Labels enter in one place, the analysis notebook, after this plan is posted.

## Data and design

Four vendors saw the same 431 images, each re-encoded once to 1024 x 1024 JPEG (q95, 4:4:4, no metadata), with the August prompts byte for byte.

| Arm | Images | Prompts | Repeats (k) | Calls per vendor |
| --- | --- | --- | --- | --- |
| Ladder | 75 eyeRounds: 15 diseases x 5 conditions (base, transformation, phone, phone_dark, motion) | free, forced | 5 | 750 |
| DR control | 200 APTOS 2019, 40 per ICDR grade 0 to 4 | neutral (= free), screening | 3 | 1,200 |
| Injection | 79 synthetic-lesion variants (81 minus 2 byte-identical duplicates) | free, forced | 5 | 790 |
| Blur | 75 synthetic directional-blur images: 15 bases x 5 anisotropy targets (5, 10, 15, 25, 40) | forced | 5 | 375 |

Models, chosen by one rule (highest-capability generally available multimodal API model per vendor on the run date, excluding compute-scaled variants; a preview only when no GA model exists at that tier): `claude-fable-5-1`, `gpt-6-astra`, `gemini-3.1-pro-preview` (no GA Pro-tier model existed), `grok-4.7`. Each vendor served one model string for the whole run.

Sampling: vendor defaults, output ceiling 16,000 tokens. Queue: one shuffle across all vendors and arms (seed 20260928), so drift and outages fall evenly.

Frozen hashes: see `prereg_manifest.json`, committed alongside this file, for the SHA256 of every input, configuration, queue, call log and library version.

## Outcomes and scoring rules

Every call is scored from its parsed JSON by fixed rules; nothing is re-scored after results are seen.

- **Forced choice:** correct when `label` equals the image's disease label exactly. A label outside the 15-class list, or an unparseable reply, is incorrect and tallied separately. The two normal classes stay distinct; collapsing them is a pre-specified sensitivity analysis.
- **Free text:** the `primary` field is mapped to a class with the August ontology (`freetext_ontology_draft.csv`, 1,370 strings). Normal adult and normal child collapse to one class. Outcome has three levels: correct, incorrect, abstain (declines to diagnose or calls the image ungradable).
- **New free-text strings:** strings the ontology does not cover are adjudicated as strings, blind to vendor, condition and correctness, before scoring. A second reader codes the highest-volume new strings plus all flagged ones; report Cohen's kappa.
- **Differentials:** top-3 accuracy (primary plus two differentials) is secondary.
- **DR screening prompt:** referable = `icdr_grade` of 2 or more. The grade governs when it disagrees with the `referable` field; disagreements are counted and reported.
- **DR neutral prompt:** mapped by the ontology to any-DR versus no-DR; a grade is scored only when the model volunteers one.
- **Self-consistency:** for each image, vendor, prompt and condition, the share of the k repeats that agree with the modal answer.
- **Confidence:** the reported `confidence` is analysed for calibration only (secondary).

## Primary hypotheses and analyses

Two primary hypotheses, both tested on the primary run's ladder arm, Holm-adjusted at a family-wise two-sided alpha of 0.05.

**H1, motion cliff.** Forced-choice accuracy under motion is lower than at base.

- Estimate: difference in accuracy (motion minus base), pooled over the four vendors and five repeats.
- Interval: 95% percentile bootstrap, 10,000 resamples of the 15 diseases (seed fixed in the notebook).
- Test: sign-flip permutation over diseases on the per-disease difference.
- Each other condition (transformation, phone, phone_dark) is compared with base the same way as secondary contrasts.

**H2, prompt format moves answers more than vendor identity.** On the same images, switching free text to forced choice changes accuracy more than switching between any two vendors.

- Model: mixed logistic regression, correct ~ prompt + vendor + condition, random intercept for disease. Free-text abstentions count as not correct.
- Statistic: |log odds ratio for prompt| minus the largest |log odds ratio| across all six vendor pairs.
- Interval: disease-level bootstrap as for H1. H2 is supported when the interval lies entirely above zero.
- Fallback: if the mixed model fails to converge, GEE with exchangeable correlation clustered by disease, stated as a deviation.

## Secondary analyses

Secondary results are reported as estimates with 95% intervals; no confirmatory claims rest on them.

1. **Self-consistency by condition:** does agreement across repeats fall before accuracy does? Disease-clustered intervals per condition.
2. **Abstention:** free-text abstention rate by condition and vendor.
3. **DR control (APTOS):**
    - Referable DR sensitivity and specificity per vendor and pooled, under each prompt.
    - Five-grade agreement: quadratic weighted kappa, exact and within-one accuracy, each shown beside the constant predictor that always answers grade 2 (20% exact, 60% within-one in this balanced set).
    - Grade-0 false-positive rate, neutral versus screening prompt, per vendor.
    - All of the above stratified by native image resolution, because resolution tracks grade in this sample.
4. **Injection arm:** forced accuracy by injection class against the 15 un-injected ladder base images. Classes come from `injection_classes.csv`; green-artifact variants are excluded here, not at collection. Normal-adult variants are reported separately because their crop differs from their base.
5. **Blur ladder:** forced accuracy across the five anisotropy targets, shown beside real motion. The ladder tops out near anisotropy 40 to 50 while real motion sits near 58, so this arm is described as an interpolation below motion, not a matched test.
6. **Per-vendor tables:** every primary and secondary estimate broken out by vendor, descriptive only.

## Reasoning sensitivity arms

The primary run is the analysis of record; the two sensitivity arms test whether its conclusions depend on how much each model reasoned.

Under vendor defaults, median reasoning tokens per call were about 1,200 for Gemini, 1,100 for Grok, 115 for Astra and 0 for Claude. So vendor differences in the primary run are partly reasoning differences.

| Setting | Claude | Astra | Gemini, Grok |
| --- | --- | --- | --- |
| Primary (defaults) | no thinking | ~115 tokens | ~1,100 to 1,200 tokens |
| Arm 05 (high) | thought on 2.4% of calls | ~363 tokens | not rerun |
| Arm 05b (max) | effort max; thought on 99.6% of calls, ~958 output tokens | effort xhigh, ~843 tokens | not rerun |

Pre-specified analysis:

1. Re-estimate H1 and the per-vendor base accuracies with the 05b responses substituted for the primary-run Claude and Astra ladder responses. Gemini and Grok keep their primary-run responses.
2. Call the primary conclusions robust if both hold: the motion-cliff estimate stays inside the primary run's 95% interval, and the rank order of vendors on base forced accuracy is unchanged.
3. Report arm 05 descriptively, including Claude's 2.4% thinking rate at effort high as a finding in its own right.
4. Report each arm's thinking rate and median reasoning tokens beside its results.

The sensitivity arms ran about 1.5 to 3.5 hours after the primary window closed, on the same locked model strings; Methods states this.

## Statistical methods

Inference is clustered at the level of independent images, never at the level of calls; resampling calls gives intervals roughly 2.5 to 2.8 times too narrow.

- **Clustering unit:** disease (15 clusters) for the ladder, injection and blur arms, since every image of a disease derives from one photograph. Individual image for APTOS, which has no patient identifiers.
- **Repeats:** analysed at the call level inside the clustered model. As a sensitivity check, each image's modal answer across its k repeats is analysed once per image.
- **Bootstrap:** 10,000 resamples, percentile intervals, one fixed seed recorded in the notebook.
- **Mixed models:** lme4 `glmer` (binomial, logit) or an equivalent. Model, formula and software version are reported.
- **Multiplicity:** Holm across H1 and H2 only. Secondary and per-vendor results carry intervals, not significance claims.
- **Software:** analysis code is versioned with the library; its hash is recorded in the results manifest.

## Exclusions and data handling

No image, call or vendor is excluded after scoring begins; every exclusion below is fixed now.

- **Vendor floor:** a vendor under 98% successful calls is excluded and named. In the primary run every vendor was at 100%, so none is excluded.
- **Unparseable replies:** scored incorrect and tallied (0 of 12,460 in the primary run; 2 recovered by the lenient parser and scored normally). In arm 05b, 5 of 750 Claude calls used the full 16,000-token ceiling while reasoning and returned no answer; they are scored incorrect under this rule, reported as a separate count, and re-analysed as missing in a sensitivity check.
- **Off-list forced labels:** scored incorrect and tallied by vendor.
- **Injection arm:** green-artifact variants excluded at analysis per `injection_classes.csv`. The two byte-identical normal-adult duplicates were dropped before collection.
- **Model drift:** any call whose served model differs from the locked string is excluded (none occurred).
- **Late changes:** any change to this plan after posting is dated and logged in the deviations table below, with its reason.

## Known limitations to report

These are stated in the paper whatever the results show.

- **Fifteen diseases, one photograph each.** Every ladder condition has one image per disease, so intervals are wide and no per-disease conclusion is drawn.
- **Public images.** EyeRounds images are likely in training data; a model may recognise an image rather than read it. Degradation may therefore partly measure loss of recognition.
- **APTOS source tracks grade.** Grade 0 is mostly 1,050 px and 550 px images; grades 2 to 4 are mostly 3,216 x 2,136 px. Normalising to 1,024 px equalises pixel count, not camera appearance.
- **No patient IDs in APTOS.** Image-level clustering may be mildly anticonservative.
- **Weaker lock for OpenAI.** The API returned an undated model name, so a silent in-place update could not be detected.
- **Gemini preview.** No GA Pro-tier model existed on the run date.
- **Reasoning asymmetry under defaults.** Addressed by the sensitivity arms, not removed from the primary run.
- **Blur ladder below motion.** Synthetic anisotropy peaked near 40 to 50 against about 58 for real motion.
- **Normal-adult injection crops** differ in geometry from their base image.
- **Single collection window.** Results describe these model versions on 28 to 29 September 2026.

## Do-not-claim list and deviations log

**Do not claim:**

- Conclusions about any single disease.
- Positive predictive value from the APTOS arm (referable prevalence is 0.60 by design).
- That the 20 to 25% ungradable figure applies to APTOS (it is an EyePACS figure).
- That synthetic blur reproduces real capture, or that the blur test matched real motion's anisotropy.
- That an anisotropy threshold generalises beyond the motion condition.
- Superiority of one vendor over another, beyond the pre-specified estimates and their intervals.

**Deviations and design changes, logged as they happened:**

| Date (UTC) | Change | Reason |
| --- | --- | --- |
| 29 Sep 2026 | Added arm 05b (Claude effort max; Astra effort xhigh) | Claude thought on only 2.4% of calls at effort high |
| 29 Sep 2026 | Claude thinking set to adaptive with effort level | API rejected the fixed-budget setting for this model |
| 29 Sep 2026 | Library 1.1.0 for sensitivity arms | Adds optional request settings; default requests byte-identical to 1.0.1 |
| 28 Sep 2026 | Concurrency raised (xAI 8, Google 6) for the full run | Throughput only; requests and records unchanged |
| 28 Sep 2026 | Library 1.0.1 before the full run | Summary function crashed on failed calls; sending and recording unchanged |
| 28 Sep 2026 | Injection class filter moved from collection to analysis | Class file unavailable; all unique variants collected instead |
