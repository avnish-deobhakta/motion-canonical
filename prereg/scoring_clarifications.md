# Motion Canonical round 2: free-text scoring clarifications

Fixed 30 September 2026 by the reviewing retina specialist (Avnish Deobhakta, MD), after string-level review blind to vendor, arm, condition and correctness, and before `truth.csv` was joined. These clarify how the August ontology and `SAP_v1.md` apply to free text; the plan did not specify them, and the August ontology applied them inconsistently.

1. **Credit is at the disease level, not the stage level.** Any answer identifying age-related macular degeneration, at any stage or in any wording, is credited as the AMD class (`intermediate_amd`). This includes unqualified "macular degeneration", exudative or non-exudative AMD, choroidal neovascularisation attributed to AMD, geographic atrophy, and disciform or fibrotic macular scars, including when the scar is reported alongside diabetic retinopathy. "Myopic macular degeneration" is myopia (rule 3), and "juvenile" or Stargardt macular degeneration is Stargardt.
2. **Only rhegmatogenous detachment is RRD.** Unqualified "retinal detachment" counts as RRD. Proliferative vitreoretinopathy counts as RRD. Tractional detachment from diabetes counts as PDR. Other tractional detachments (ROP, toxocariasis, haemangioblastoma) and exudative or serous detachments (Coats, tumours) are `other_named`. Familial exudative vitreoretinopathy is `other_named`.
3. **Myopia counts as the myopic staphyloma class** even without the word "staphyloma", unless the answer commits to a different diagnosis and mentions myopia only incidentally (for example retinitis pigmentosa or choroidal melanoma in a myopic eye, or myelinated nerve fibres on a tessellated fundus).
4. **Hemiretinal (hemispheric) vein occlusion counts as BRVO.**
5. **Abstain means no committed diagnosis.** "Poor quality, but likely X" is scored as X.

`scoring_overrides.csv` applies these rules to 43 August ontology mappings they contradict, and corrects 2 slips in the clinician review (rows 14 and 239 of `review_queue`). The review itself (324 required rows and 2 August conflicts) is fingerprinted by SHA256 of its sorted `raw_response` and `adjudicated_class` pairs: `5384f7cd7757bc9224458b784a6a3df5d523a8d01710de0ed4a43b0256db59e8`.

In reporting, the AMD class is labelled "AMD (any stage)".
