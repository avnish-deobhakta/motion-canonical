# Statistical analysis plan, LLM as second reader (llm2r), v1

Status: v1, signed off by all coauthors on 23 October 2026, registered by the lock cell of notebook llm2r_01_audit_and_sample_v4 before any LLM read. Content identical to v0.2 of 23 October 2026 (which revised v0 of 22 October; revisions are marked "v0.2"). Any later change is an amendment with its own dated file.

Study short name: llm2r. Separate from the npj resubmission (second CNN reader trained on different data) and from DR6 (six-dataset multimodal model evaluation). It reuses the locked first readers from npj and the reading pipeline, prompts and Jev scoring stage from DR6.

## 1. Question

In a screening pipeline where a purpose-built CNN reads every fundus photograph, does adding a general-purpose multimodal model as a second, independent reader catch more of the CNN's missed severe disease (severe NPDR or PDR) than adding a second CNN, when both pipelines refer the same number of patients?

## 2. Decisions fixed in this plan

These five were the open questions in the 21 October working note. Each is settled here with its reason, so that the choice is on record before any read.

2.1 Primary partner: one general-purpose model, Gemini (model string fixed in models.json at lock). Reason: the miss-rate rule says a partner's own miss rate is the binding constraint, and Gemini had the highest sensitivity in DR6 (96.0% on the common set, 87% or more on EyePACS). The other models (GPT-6, Grok, Claude, Qwen) are secondary partners reported with the identical analysis and no multiplicity adjustment, labelled as secondary. The power simulation (section 8) does not support a four-way primary with correction.

2.2 Primary score for the LLM partner: Jev P(referable) computed from the model's free-text description, pinned to snapshot jev-1.14-20261014. Sensitivity analysis: mean of the three forced-choice repeat grades (ties broken at random with a fixed seed), which does not use Jev. Reason: the matched-volume comparison needs a continuous rank; Jev's probability is continuous and calibrated, and DR6 showed Jev reproduces a specialist's grade of the same text (92.7% agreement, QWK 0.96).

2.3 Primary design: full parallel reading. Both readers score every eye in the analysis sample; the pair ranks by the higher of the two readers' percentiles and refers the top f (the npj matched-workload rule, which is a truncated union, not an OR gate). "Second look at negatives" (the LLM reads only the CNN's negatives) is a secondary, deployment-oriented analysis computed from the same reads, since the CNN's negatives are a subset of the sample; it needs no extra API calls.

2.4 DR6 reads are not reused in the primary analysis. All primary reads come from a fresh sample of the held-out sets (section 4). Reason: the DR6 sample is 240 EyePACS and 340 DDR images enriched by grade, drawn without reference to the npj splits, under a plan written for a different question. DR6 reads are used only for (a) the contamination and overlap audit and (b) a pilot estimate of each partner's miss rate, reported as pilot.

2.5 Foundation-model rung: RETFound is included as a secondary comparison partner (inference only, from the npj locked checkpoints if one trained on different data from the first reader exists; otherwise omitted and said so). It completes the ladder same data, different data, different pretraining, general-purpose model. It is not part of the primary comparison.

## 3. Readers

First readers (A): the npj locked CNNs, no retraining: the EyePACS-trained reader and the DDR-trained reader. Model tags and checkpoint hashes recorded in models.json at lock.

Comparison partner (B_cnn): for each first reader, the best non-sibling CNN partner as defined in the npj plan (different training source, selected on validation, not on the confirmation set). Sibling CNN (same training source and starting checkpoint) is reported as a secondary partner for the ladder.

LLM partners (B_llm): Gemini (primary); GPT-6, Grok, Claude, Qwen (secondary). Reading pipeline, image normalisation, description prompt and forced-choice prompt are the DR6 ones, verbatim, hashes recorded at lock. One description call and three forced-choice calls per image, as in DR6. Jev call on the description.

Ungradable: if a partner's description is scored by Jev as ungradable with probability above the referable probability, the eye is treated as referred by that partner (it consumes a referral slot). Reason: this is what a screening program does with an ungradable read, and it is conservative against the LLM at matched volume. The count of such eyes is reported.

## 4. Analysis sample (v0.2: rewritten from the actual npj v2 layout and the sample drawn on 22 October 2026)

Populations. Two held-out populations, neither used for training, checkpoint selection or the npj partner analysis:
- EyePACS: the npj fresh set (15,794 eyes, all from the Kaggle TRAIN partition and never trained on) together with the lock v4 confirmation pool of the Kaggle TEST partition (17,858 eyes). Total 33,652 eyes, 16,826 patients, 1,484 severe (712 and 772 by stratum), referable prevalence 0.195. The Kaggle-test estimation pool is excluded because the npj partner analysis was estimated on it. Stratum is recorded.
- DDR: the DDR-trained readers' held-out test split, 3,725 images in 143 pseudo-patient groups, 346 severe (275 PDR), referable prevalence 0.454. This is the only part of DDR on which a DDR-trained partner's score is uncontaminated. The DDR-trained train/val partition is recorded only through the frozen prediction files and is derived from them.

Sampling frame per population: all severe eyes (grade 3 or 4) plus a simple random sample of 1,500 non-severe eyes drawn with seed llm2r-2026-10-22. Weights: 1 for severe; N_ns / n_ns for sampled non-severe (21.45 in EyePACS, 2.25 in DDR). Drawn sample: 2,984 EyePACS images and 1,846 DDR images. All weighted estimates use these weights; the referral cutoff for a reader is the weighted (1 minus f) quantile of its score.

Images given to the LLM partners: the original photographs normalised to 1024 x 1024 by the DR6 pipeline for the fresh stratum (Kaggle TRAIN archive) and for DDR; the npj 512-pixel preprocessed images, upscaled by the same normaliser, for the Kaggle-test confirmation stratum, for which no originals are on Drive. Image source is recorded per eye and LLM sensitivity is reported by stratum (Table 4) as a resolution check. CNN scores are the frozen npj float64 predictions, made from the 512-pixel images; DDR-trained models had never been scored on EyePACS, so notebook 01b scored them on the sampled EyePACS images only, after reproducing the frozen DDR predictions on 300 held-out DDR images (r = 1.000 for DenseNet; RETFound loader validated the same way).

Readers (architecture held fixed across the ladder). First readers: EyePACS-trained DenseNet121 5-class (A_eyepacs) and DDR-trained DenseNet121 5-class (A_ddr), npj replicate 0. For each first reader: sibling partner = same source, same architecture, replicate 1; non-sibling partner = the other source, same architecture; pretraining rung = RETFound 5-class of the same source.

Operating points (v0.2). EyePACS: the npj rule, f = 1.31 x referable prevalence = 0.255. DDR: a fixed referral burden f = 0.25, the secondary operating point of the npj DDR analysis. Reason: at the npj rule DDR's f would be 0.594, at which the first readers miss 0 and 9 severe eyes and there is nothing for a partner to rescue; at f = 0.25 they miss 33 and 77. The operating point is fixed per population before any LLM read and recorded in the lock manifest.

Severe misses by the first readers at these operating points, which the primary analysis is conditioned on: A_eyepacs on EyePACS 88 of 1,484; A_ddr on EyePACS 171 of 1,484; A_eyepacs on DDR 77 of 346; A_ddr on DDR 33 of 346.

## 5. Outcomes

Primary: for each first reader A and each held-out set, the rescue fraction R(B) = (number of A's severe misses referred by the pair A+B at matched volume) / (number of A's severe misses), compared between B = Gemini and B = best non-sibling CNN on the same misses. Estimand: the difference R(Gemini) minus R(B_cnn), pooled over the four cells (two first readers by two populations), each cell at its own pre-specified operating point (section 4). v0.2: the EyePACS-only and DDR-only pooled differences are reported alongside, since the two populations differ in operating point and image source.

Secondary:
- Net severe misses of the pair at matched volume (the npj outcome), A+Gemini against A+B_cnn.
- The same for each secondary partner and for the ladder (sibling, non-sibling, RETFound, each LLM).
- Shared-miss ratio for each A+B pair, and the rescue predicted by the miss-rate rule, 1 minus (shared-miss ratio times B's miss rate), against the observed rescue.
- Second look at negatives: rescue when the LLM reads only A's negatives and the pair's volume is matched by reranking the negatives.
- Sensitivity: primary analysis repeated with the mean repeat grade score instead of Jev.

## 6. Inference

Unit of analysis: the severe eye. A's severe misses from the two first readers overlap on the same eyes, so the pooled estimate uses the union of missed eyes with cell as a stratum, and confidence intervals come from a patient-level bootstrap (2,000 resamples, seed fixed at lock) that resamples patients, not eyes. The paired comparison within a cell is an exact test on discordant pairs (McNemar). Two-sided alpha 0.05 for the single primary comparison. Secondary results are reported with intervals and no multiplicity adjustment, labelled secondary.

Cells with fewer than 10 severe misses by A are reported as inconclusive, as in npj.

## 7. Contamination and overlap audit (before sampling)

Run once, aggregate output only:
1. For each DR6 EyePACS and DDR image, whether its source stem or image hash appears in any npj train or validation split (contamination) and whether it appears in the EyePACS confirmation set or the DDR test set (overlap). Report counts by dataset and grade band only.
2. Any contaminated image is excluded from all pilot use.
3. Record the audit output hash in the lock manifest.

## 8. Power (v0.2: recomputed from the drawn sample)

Simulation (llm2r_power.py, 23 October 2026), assuming the CNN partner rescues 42% of a first reader's misses (npj fresh-set value), an exact test on discordant misses, two-sided alpha 0.05, 80% power, correlation 0.1 between the two partners' rescue indicators:

- A_eyepacs on EyePACS (88 misses): detectable difference about 0.22.
- A_ddr on EyePACS (171 misses): about 0.16.
- EyePACS cells pooled (up to 259 misses; the two readers' misses overlap on some eyes): about 0.12.
- A_eyepacs on DDR (77 misses): about 0.23. A_ddr on DDR (33 misses): about 0.35; this cell is reported but is underpowered on its own.
- All four cells pooled (369 misses before overlap, roughly 300 after): about 0.11.

So the primary comparison can detect a moderate advantage or disadvantage for the LLM partner (a rescue fraction of about 0.53 or more, or about 0.31 or less, against the CNN partner's 0.42). It cannot resolve differences under 0.1, and the ladder (section 5) is descriptive. Stated in the paper as a limitation decided in advance.

## 9. Cost, drift and run-once guards

- Model strings for every API model, the Qwen weights hash, the Jev snapshot, the two prompts, dr6_lib version, the sampling seed, the CNN reader tags and the per-population operating points are written to models.json and lock_manifest.json at lock and committed.
- Every read is keyed by (image_id, model, call type, repeat) and written once; a rerun that would overwrite an existing read is refused.
- All reads for a partner are completed within a single calendar week where possible; dates recorded.
- BRSET is excluded from this study (images cannot go to any API). A Qwen-only BRSET arm is deferred to a separate plan.
- Conflict-of-interest statement on TypeSafe AI (Jev) to be written at lock.

## 10. Files

- Plan: this file, llm2r/prereg/SAP_llm2r_v1.md.
- Notebooks: llm2r_01_audit_and_sample_v4 (audit, sample draw, CNN scores, lock), llm2r_01b_cnn_inference (DDR-trained checkpoints on the EyePACS sample), llm2r_02_reads (LLM reads), llm2r_03_jev_and_freeze (Jev scoring, scores table), llm2r_04_analysis (matched volume, rescue, ladder).
- Data: llm2r/sets/registry.csv (anonymised image ids, no grades), llm2r/_KEY_DO_NOT_SHIP/key.csv (source stems, grades, patient ids, weights), llm2r/runs/, llm2r/results/.
