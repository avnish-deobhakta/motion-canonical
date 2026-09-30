# Motion Canonical round 2: analysis plan addendum, consumer chat-interface arm

Version 1, fixed 30 September 2026, before any chat-interface task was run · Avnish Deobhakta, MD

## Purpose

The API arms measure the models. Clinicians use the consumer chat apps, which add their own system prompts, image handling and model routing. This arm measures what the apps return for the same images and prompts, so the difference between app and API answers can be estimated with the model held constant.

## Design

- **Images:** the 30 ladder images at base and motion for all 15 diseases, byte-identical to the files the API received (1024 x 1024 JPEG), renamed to neutral file names `I01` to `I30` in a random order. The operator sees no image identifier or label.
- **Prompts:** the free and forced prompts, byte-identical to the API arms.
- **Products:** the ChatGPT, Claude, Gemini and Grok consumer apps, each set to the same model used through the API (`gpt-6-astra`, `claude-fable-5-1`, `gemini-3.1-pro-preview`, `grok-4.7`) using the app's model picker. Automatic routing is never used. The model name the app displays is recorded on every task.
- **Repeats:** 2 per image, prompt and product; repeat 1 on days 1 and 2, repeat 2 on days 3 and 4, so no pair of repeats shares a day.
- **Tasks:** 30 images x 2 prompts x 4 products x 2 repeats = 480, in product blocks of 30 per day with the block order rotated across days and task order randomised within blocks (seed 20260930).
- **Session hygiene:** a new chat for every task; the app's temporary or incognito mode where offered, otherwise chat history and memory off; custom instructions, web search and tools off. The image is uploaded as a file, never as a screenshot.
- **Capture:** the full reply is pasted verbatim, with the displayed model name and time. A refusal or non-answer is recorded as such and never retried.
- **Operator:** a single operator (the study clinician), who records replies without judging them.
- **Window:** all tasks within 7 days of the first.

## Scoring

The same pipeline as the API arms: forced replies by exact label; free-text replies through the frozen class map, with any new strings sent to blind clinician review under the committed scoring clarifications before scoring. Replies that are not the requested JSON are scored from their text by the same rules; a reply that names no diagnosis is a non-answer.

## Estimands and analysis

1. **Primary:** the paired difference in forced-choice accuracy, app minus API, pooled over vendors, separately at base and at motion. The API comparator is the primary run's five repeats for the same image, vendor and prompt. The 95% interval comes from 10,000 bootstrap resamples of the 15 diseases.
2. The same difference per vendor.
3. The motion effect (motion minus base) within the apps, estimated as for H1 in `SAP_v1.md`.
4. **Secondary:** the free-text accuracy difference; abstention and refusal rates; how often the displayed model differs from the requested one; agreement between the two app repeats; agreement between each app answer and the modal API answer for the same image.

These are estimates with intervals. No confirmatory hypothesis is tested in this arm, and it does not alter H1 or H2.

## Exclusions and deviations

- A task where the app displayed a model other than the requested one is kept, flagged, and analysed both with and without such tasks.
- A task not completed within the window is reported as missing; it is not re-run later.
- Any change to this protocol after the first task is logged here with its date and reason.
