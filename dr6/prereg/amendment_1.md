# DR6 amendment 1: parsing of Qwen replies (17 October 2026, after unblinding)

**Status.** Written after the key was opened (results commit 4d9ddba0, 17 October 2026). This is a deviation from the registered plan and is reported as such. The registered results are kept unchanged in `results/results_dr6.json`; the amended results are reported alongside them in `results/results_dr6_amend1.json`.

**What was found.** The results showed implausible values for the open-weight model only: Qwen self-pass sensitivity 0.14, and the fixed rule calling 100% of Qwen descriptions referable. A read-only diagnostic (counts only) showed the cause. Qwen3.8 writes its reasoning first and ends it with a closing `</think>` tag, but emits no opening `<think>` tag (6,360 of 6,372 stage-1 replies and 1,593 of 1,593 self-pass replies had the closing tag; none had an opening tag). The registered library removes reasoning only when both tags are present. The JSON parser therefore searched the reasoning, where Qwen quotes the example from the prompt (`{"icdr_grade":0,...}`), and returned grade 0 instead of the final answer.

**Effect, measured before any re-scoring.**
* Stage-1 forced choice: 292 of 4,779 Qwen grades change (261 registered 0 become "ungradable").
* Self-pass: 912 of 1,593 Qwen grades change (905 of them registered 0).
* Fixed rule: 742 of 1,593 Qwen descriptions change grade.
* API models: the change applies only to text containing the tag; the amended run checks that their primary results are identical to the registered run.

**The change.** For text containing `</think>`, everything up to and including the last `</think>` is removed before parsing (forced choice and self-pass) and before the fixed rule is applied. A Qwen reply without the closing tag (output ceiling reached during reasoning; 12 stage-1 replies) has no final answer and is treated as unreadable, which the plan counts as a referral. Nothing else changes: same data, same frozen responses, same weights, bootstrap, seed and outcome definitions. The code is `scoring_dr6_26_amend1.py`, which differs from the registered scoring code only in these lines.

**Not changed.** Jev received the full Qwen text, reasoning included, because the stage-2 queue used the same reasoning filter; the human grader saw the same text. Jev's answers and the human grades are kept as they are, since both read identical input. No model calls are repeated.

**Reporting.** Primary and key secondary results are reported as registered and as amended. Conclusions that depend on Qwen are drawn from the amended analysis and labelled as such. Results for the four API models are unaffected, which the amended run checks.
