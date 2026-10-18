# DR6 amendment 2: image-description audit of missed referable eyes (17 October 2026, after unblinding)

**Status.** Post hoc. Written after the key was opened and after the amended results (commit f6467c67) and the exploratory referral-rate analysis (notebook dr6_29) were seen. It is registered, with the case sample's hash, before the review packet is opened by the grader. It is reported as an exploratory analysis and changes no registered or amended result.

**Question.** When a model missed a referable eye, was the lesion visible on the image but absent from the model's description (a perception miss), described but graded below referable (an interpretation miss), or was referable disease not visible to a retina specialist on the prepared image (a reference grade the image does not support)?

**Sampling frame.** Pairs of (model, image) from the common set (gradable images from APTOS 2019, DDR, EyePACS, IDRiD and SUSTech-SYSU; BRSET is not included) where:
* the reference grade is 2, 3 or 4;
* at least 2 of the model's 3 forced-choice repeats were grade 0 or 1 (a miss; "ungradable" counts as a referral and so is not a miss);
* the model's free-text description is readable (for Qwen3.8-27B, a closing reasoning tag is present; the description shown is the text after the last closing tag, as in amendment 1).

**Sample.** 60 cases in two strata of 30: EyePACS, and the four other public datasets pooled. Within each stratum, 6 cases per model. Candidates are ordered by a seeded shuffle (seed string "20261016|audit2|<stratum>|<model>"); cases are taken in that order, skipping an image already taken in the stratum, so each image appears at most once per stratum. If a model has fewer than 6 eligible cases in a stratum, the shortfall is filled from the remaining candidates of the other models in that stratum, in seeded order. No case is chosen or replaced by eye.

**Review.** One retina specialist (A.D.) reviews each case: the prepared image (1024 x 1024 pixels, the bytes every model received) beside the model's description. Cases are in random order. The packet does not show the model, dataset, reference grade or the model's grades. The grader knows the sampling frame (all cases are referable by the reference and were missed by the model); this is a limitation. For each lesion type the grader enters one code:
* `0`: not visible and not described;
* `V`: visible on the image, not described;
* `D`: visible and described;
* `X`: described, not visible.

Lesion types: microaneurysms (MA), intraretinal haemorrhages (HEM), hard exudates (HE), cotton wool spots (CWS), venous beading (VB), intraretinal microvascular abnormalities (IRMA), new vessels (NV), preretinal or vitreous haemorrhage (PVH), laser scars (LASER). The grader also records image quality (`good`, `limited`, `ungradable`) and their own ICDR grade from the image (`0` to `4` or `ungradable`). Laser scars without other findings are graded 2, as in the text-grading arm.

**Classification of each case (applied in this order, by code, not by judgment).**
1. *Ungradable on review*: grader grade is `ungradable`.
2. *Reference not supported*: grader grade is 0 or 1.
3. *Perception miss*: grader grade 2 to 4, and at least one lesion other than MA is coded `V`.
4. *Interpretation miss*: grader grade 2 to 4, and no lesion other than MA is coded `V` (the description named the findings that make the eye referable, and the model still graded it below 2).

**Outcomes (descriptive; no tests).**
* Share of cases in each class, overall and by stratum, with Wilson 95% confidence intervals.
* EyePACS minus other datasets for the share "reference not supported" and for "perception miss", with Newcombe 95% confidence intervals.
* Lesion-level description sensitivity: D / (D + V) for each lesion type, pooled; and the number of `X` codes (lesions described but not visible).

**Integrity.** Before the packet is opened, the notebook commits this amendment and a manifest with the SHA-256 of the case key (case to model, image, dataset and reference grade), the packet and the blank grading sheet. The scoring step checks those hashes before reading the completed sheet. The completed sheet is committed with the results.

**Limitations to be reported.** Single grader, who is an investigator and also graded the text arm; no second reader. The grader is not masked to the sampling frame. Post hoc analysis. Sixty cases give 95% confidence intervals of about plus or minus 13 percentage points overall and about plus or minus 18 points within a stratum.
