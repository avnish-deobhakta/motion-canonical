# llm2r-ext: external confirmation on DeepDRiD. Statistical analysis plan v1

Status: version 1, signed off by the coauthors on 28 October 2026 and locked by notebook 08 before any LLM output on DeepDRiD existed. Drafted 27 October 2026.

## 1. Why this amendment exists

The llm2r primary analysis (plan v1, locked 23 October 2026) was negative: at matched referral volume, Gemini scored through Jev with the ungradable rule rescued fewer of a CNN first reader's severe misses than a second CNN did (pooled difference -0.211, 95% CI -0.332 to -0.013). Post hoc analyses (notebooks 05 v2, 06 and 07) generated a specific hypothesis. When each LLM is scored on its own forced-choice grades, and when no in-distribution CNN is available, the LLM partner rescues more severe misses than a second CNN does. This held for four of five LLMs against an out-of-distribution CNN. This amendment tests that hypothesis on a population that played no part in generating it.

## 2. Population

- **Dataset:** DeepDRiD regular fundus images, all 2,000: training 1,200, validation 400 and Online Challenge 1 test 400. These come from 500 patients, with two photographs per eye. Ultra-widefield images are excluded as a different modality.
- **Labels:** the eye-level DR grade published with each image (left or right eye column; Challenge 1 from `Challenge1_labels.xlsx`). Severe means grade 3 or 4 (466 images). Referable means grade 2 or above (864 images, 43%).
- **Patient:** the patient identifier, recovered from the image identifier prefix and checked against the `patient_id` column where present.
- **Independence:** DeepDRiD was not used in llm2r, DR6 or the npj analyses, and none of the CNN readers was trained on it. On this population every CNN reader is out of distribution.
- **Sampling:** none. Every image is read, with weight 1.

## 3. Operating point

A fixed referral fraction f = 0.25. This is the rule already pre-specified for DDR in llm2r v1, because the npj rule (1.31 × referable prevalence) would exceed 0.5 here and leave the CNNs almost no severe misses.

## 4. Readers

- **First readers (A):** DenseNet121 five-class, EyePACS-trained and DDR-trained (the llm2r first readers), frozen checkpoints from the npj v2 run.
- **CNN comparators for each first reader:**
  - the other-source DenseNet121 (non-sibling, the primary comparator);
  - the same-source seed-1 DenseNet121 (sibling);
  - the same-source RETFound (`retfound_5class`).
- **Image source:**
  - CNN scores use the DeepDRiD processed_v2 images with the npj transform (resize 256, centre crop 224, ImageNet normalisation; referral probability = softmax mass on grades 2 to 4). Each checkpoint must first reproduce its frozen npj predictions on 300 images from its own source (Pearson r > 0.999, 99th percentile of absolute difference < 0.01, maximum < 0.05) or the notebook stops. This rule replaced an earlier single-maximum rule (max < 0.02) before lock, because one image at 0.0208 failed it while r was 1.0000, which is GPU numerical noise rather than a wrong model. All six checkpoints pass (r = 1.0000; 99th percentile 0.003 to 0.006; maximum 0.004 to 0.021); the log is `runs/cnn_inference/selfcheck_log.csv`.
  - The DeepDRiD and DDR processed_v2 images are both 512 × 512 RGB PNG. The black-border fraction is 0.32 for DeepDRiD and 0.22 for DDR, so the DeepDRiD retinal disc fills slightly less of the frame (about 93% of the width against a fully inscribed circle). This is reported as a property of the source images; the preprocessing is not tuned for DeepDRiD.
  - LLM reads use the original photographs, normalised to 1,024 px by `dr6_lib.normalise_image`, as in llm2r.
- **LLMs:** the five llm2r models with the exact strings locked in llm2r `models.json`:
  - `gemini-4-argon` (primary);
  - `gpt-6-astra`, `grok-4.7`, `claude-fable-5-1` and `Qwen/Qwen3.8-27B` at the pinned revision (secondary family).
- **Prompts:** the llm2r prompts, verbatim: one free-text description and three forced-choice grades per image.

## 5. Scores

- **Primary LLM score:** the mean of the three forced-choice grades, with ungradable coded 5 and a seeded random tie-break of at most 1e-6. Fewer than three parseable repeats: the mean of those available. None parseable: the image is ranked as ungradable (5).
- **Secondary LLM scores:** the llm2r locked score (Jev P(referable), with the ungradable rule) and raw Jev P(referable). Each Jev score takes the same seeded tie-break. Jev snapshot `typesafe/jev-1.14-20261014`.

## 6. Hypotheses and tests

The matched-volume pair, the rescue fraction, the patient-level bootstrap (2,000 resamples, seed 20261022) and the exact McNemar test are as in llm2r v1 section 6.

- **H1 (primary).** Pooled over the two first readers, Gemini's rescue minus the non-sibling CNN's rescue is greater than zero. Supported if the two-sided 95% bootstrap CI lies above zero.
- **H2 (gated, tested only if H1 is supported).** Severe visual loss avoided per million eyes, Gemini against the non-sibling CNN, pooled over the two first readers, is greater than zero (95% CI above zero). Uses the notebook 06 harm model with its parameters unchanged: one-year delay; 4.75% added risk of severe visual loss per missed PDR eye; 15% one-year progression of severe NPDR to high-risk PDR; 0.2% direct severe-NPDR risk; 15 sight-years per event. DeepDRiD's case mix is enriched, so H2 is reported per 10,000 eyes of this sample and not as a population estimate.
- **H3 (secondary family).** The H1 test repeated for GPT-6, Grok, Claude and Qwen, with Holm adjustment across the four. The class-level claim is made only for the models whose adjusted test is supported.
- **Descriptive.** The sibling CNN and RETFound comparators; the locked and raw-Jev scores; each LLM's ungradable rate (Jev and forced-choice); the miss-rate rule (observed vs predicted rescue); results by first reader.

## 7. Minimum information

- A first-reader cell with fewer than 10 severe misses is reported but dropped from the pooled tests.
- If the pooled first-reader misses number fewer than 40, H1 to H3 are reported as descriptive only.
- The miss counts come from the CNN scores, computed and printed before lock (they involve no LLM output). Observed: the EyePACS-trained first reader misses 144 of 466 severe images (83 patients) and the DDR-trained first reader misses 161 (87 patients), 305 pooled. Both cells clear the minimum, so H1 to H3 are confirmatory.

## 8. Order of work and lock

1. Notebook 08:
    - build the registry and key;
    - write the 1,024-px images for the LLMs;
    - run the CNN self-checks and inference;
    - print the first-reader miss counts;
    - lock this plan, the registry and `models.json` to the repository (`llm2r_ext/prereg/`).
2. Notebook 09: LLM reads and the Jev pass, then the freeze.
3. Notebook 10: analysis as specified here.

No LLM output on DeepDRiD exists or is opened before the lock.

## 9. Deviations from llm2r v1

- Population: DeepDRiD, read in full, instead of a sample.
- Operating point: f = 0.25 throughout.
- The primary score is the forced-choice mean rather than Jev; this is the hypothesis under test.
- The secondary LLMs form a Holm-adjusted confirmatory family rather than descriptive rows.
- H2 is added as a gated outcome.

## 10. Disclosure

The hypothesis was generated by post hoc analyses of llm2r data (notebooks 05 v2, 06, 07), as stated in section 1. DeepDRiD was chosen after those analyses, because it was the largest available set with patient identifiers that none of the study's readers or analyses had touched, and before any LLM output on it existed.
