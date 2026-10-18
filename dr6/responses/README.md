# DR6 model responses (public datasets)

Frozen model responses for the five public datasets (APTOS 2019, DDR, EyePACS, IDRiD, SUSTech-SYSU).
BRSET records are removed: BRSET is a PhysioNet credentialed dataset and responses about its images are not redistributed.

Files (gzipped JSON lines, one record per call):
* stage1_api.jsonl.gz: Claude, GPT, Gemini and Grok image calls (forced choice, 3 repeats; free text, 1 repeat)
* stage1_qwen.jsonl.gz: Qwen3.8-27B image calls
* stage2.jsonl.gz: text-grading calls (Jev and self-pass) on the API models' descriptions
* stage2_qwen.jsonl.gz: text-grading calls on Qwen's descriptions
* stage2_jev_image.jsonl.gz: Jev image calls, if that arm was run

Each record keeps every field the runner logged except `raw` (the provider's full response envelope), which the scoring code does not read.
No image bytes are in any record; `image_sha` is the first 16 hex characters of the SHA-256 of the prepared JPEG the model received.
`responses_manifest.json` gives, for each file, the SHA-256 of the frozen source file (which matches freeze_stage1.json or freeze_stage2.json),
the number of records read, removed as BRSET and kept, and the SHA-256 of the committed file.
Results for the public datasets can be recomputed from these files with prereg/scoring_dr6_26_amend1.py; BRSET results cannot.
