---
title: Work With Projects Using Koharu Agent
description: Ask the agent to inspect pages, edit semantic content, organize layers, and run the pipeline.
---

# Work With Projects Using Koharu Agent

The Agent works against the currently open project. If none is open, use the Agent button on the project screen and ask it to create or open one. Give it a concrete goal and enough scope to identify the affected pages or text.

## What the agent can do

The current host tools let the agent:

- create a uniquely named project or open an existing project in Koharu's project library;
- import images, archives, or PDFs from explicit absolute file paths without a native dialog;
- inspect the latest semantic project state without page images;
- render and view a specific page when appearance matters;
- rename, reorder, or delete pages;
- add a paragraph text box;
- edit source text, translation, typography, geometry, visibility, and opacity;
- move or delete elements;
- run the configured pipeline for the project, selected pages, or selected text elements;
- export explicit page IDs, or all pages, as PNG or PSD files to an existing absolute directory without a native dialog.

The headless E2E host begins with `run_source_analysis`, which runs detection and OCR without translation or inpainting. It then analyzes the unchanged original raster for each detected source-text box and exposes a panel candidate only when a dark connected contour closes on all four sides, encloses the source box inside its safe interior, has a materially distinct interior, stays off the page edge, and leaves room for the 12 px minimum font plus 4 px per-edge clearance. When exactly one required detector free-text element and exactly one such panel share that recorded source identity, the host immediately assigns the UI role and binds the element to the detector-recorded safe interior; dialogue, captions, SFX, other roles, existing target containers, and ambiguous source/panel matches remain untouched. Full contour, border, raster, scene, and host-binding evidence is written to content-addressed run artifacts; model results and traces retain the artifact path, media type, byte length, BLAKE3 digest, concise bounds, confidence, source relationship, and candidate/rejection summary. For each page, `inspect_source_evidence` plus `view_page_source_debug` then exposes the authoritative source pixels. `classify_difficult_sfx` may opt out only detector free-text that is positively classified as decorative sound-effect lettering and difficult to OCR. Each decision binds the exact revision, ordinal, element, crop digest, and debug label; records classifier, confidence, rationale, and separate evidence for visual form, page function, OCR difficulty, and exclusion of required content; and leaves the source region/OCR/crop in evidence as `skipped_difficult_sfx` with no target translation or visible target render owner. Dialogue/container-linked text, known caption/UI roles, ambiguous evidence, confidence below 0.90, and generic free text are rejected. Text length and script are never classification evidence. `verify_ui_panel_anchor` is exposed only for detector evidence that remains genuinely unbound; it retains the same exact finite panel, source containment, UI role/function, safe-interior, provenance, confidence, and anti-nearest-selection gates. `run_pipeline` then preprocesses and translates all required content.

The host also exposes a narrow page-translation repair sequence. Before the initial `review_pages`, a semantic batch may be authorized by the matching four-part page-evidence set below. Evidence must be refreshed after any intervening scene mutation. After `review_pages`, its deterministic repair state is binding and each permitted repair follows the same evidence and re-review loop:

1. `inspect_page_evidence` atomically writes and returns the page-scoped source dossier, full original page and ordinal crops, source-debug overlay, translated dossier, and translated-render debug overlay. It preserves deterministic reading order, exact page/revision provenance, accepted predecessor-page context, and the four existing BLAKE3 bindings. Original pixels remain authoritative and OCR remains fallible.
2. `revise_page_translation` requires the exact current `scene_revision` and all four fresh matching dossier/artifact BLAKE3 digests from that bundle. It atomically applies only nonempty, correctly language-tagged source/translation edits belonging to that page. It rejects stale, missing, digest-mismatched, or cross-page evidence as well as geometry, visibility, opacity, region ownership, padding, typography, and layout fields. A source correction cannot be authorized by OCR text alone. The trace retains all four evidence digests and their page/revision provenance together with first OCR and per-element before/after state.
3. `preview_text_layout` and `commit_text_layout` control Korean keep-words line breaking, maximum lines, alignment, font-size decrease, and positive safe-padding increases without accepting raw bounds. For positively verified required UI or adjacent free-dialogue text, this is also the only path that may preview replacing exact source-bound placement with the verified target relation. Preview rasterizes the candidate and rechecks semantic ownership, finite positive visible rendering, page containment, translated-text overlap safety, and strict improvement of the blocking condition; stale, unverified, unsafe, or non-improving candidates never commit. Font, glyph, anchor, and contour-clearance measurements remain diagnostic and do not authorize a repair.
4. `review_pages` writes the current original/rendered/semantic bundle and re-runs deterministic acceptance. If an optional command judge is configured, it also executes that judge and its independent decision remains authoritative. Otherwise the bundle stays pending in the current agent loop.
5. Without a command judge, each outstanding page exposes exactly one `inspect_page_evidence` obligation followed by `submit_visual_semantic_review`. Submission requires the exact current page/revision plus the fresh BLAKE3 bindings from all four bundled artifacts. Its structured decision includes all seven required booleans, a summary, and issues. The reviewer must compare actual source pixels, translated meaning and render, target-language naturalness, reading order, completeness/duplicates, typography/layout, and whether every skip is truly difficult decorative SFX with no dialogue, caption, UI, or general free text skipped. An incorrect or unjustified skip rejects the review and blocks export; deterministic geometry is not semantic evidence. Every bundled page must be accepted before export. With a command judge configured, this tool cannot replace or bypass it.

An active deterministic repair plan remains binding: fresh page evidence cannot bypass it. A semantic page batch can edit only the first semantic failure and its allowed semantic field, while an actual blocking layout plan blocks semantic mutation and exposes its own preview action. An accepted or failed visual/semantic review cannot authorize another repair, and evidence alone never accepts review or opens the export gate. Export requires both deterministic acceptance and the configured command-judge decision, or accepted page-bound in-loop decisions when no judge is configured.

Source-region geometry always remains the identity and evidence boundary for OCR and source semantics. It is not necessarily the correct typography frame after a writing-mode transformation or for required text drawn inside an in-world UI. For a verified vertical-Japanese to horizontal-Korean transformation, the acceptance record identifies the applicable source or committed target anchor and reports its containment, coverage, fitting, font, glyph, and contour-clearance measurements. Those post-hoc measurements remain diagnostics rather than export rejection gates. Required dialogue, captions, and UI still need nonempty source and target-language semantics, successful finite positive visible rendering, page containment, non-obscuring translated-text placement, and semantic review. A detector-backed free-dialogue candidate remains source-bound and cannot become the authoritative text-safe target or move rendered text until the exact controlled preview succeeds and is committed. Positively verified required UI text may likewise use the inset interior of its exact detected finite panel after its visual-semantic decision and controlled relation migration. Without the relevant committed association, the source region remains the layout anchor; no nearest-panel, arbitrary geometry expansion, source-box-size inference, or UI-enabled SFX skip is allowed.

It cannot manage provider credentials or read arbitrary files through these tools. Imported paths must be absolute, exist, be regular supported files, and exports require an existing absolute directory.

## Write a useful request

Name the scope, desired result, and constraints:

> Review pages 3–5. Correct obvious OCR errors, translate into natural English while preserving honorifics, and do not change typography.

For visual work, say what should be inspected:

> Check whether the text on page 8 overflows its bubble. Adjust only size and line breaks.

The agent chooses page rendering only when needed; semantic inspection is cheaper and more private for text-only tasks.

## Mutations and history

Agent edits use the same project operations as the desktop UI. Successful changes update the scene, renderer, project revision, and undo history rather than writing a separate agent document.

An agent pipeline run commits completed stages incrementally and supports cooperative cancellation. Canceling the chat request also asks an active agent-started pipeline to stop.

## Review the result

After the agent finishes:

1. inspect changed pages on the canvas;
2. verify source and translated text;
3. check geometry and typography at reading size;
4. use undo for a coherent unwanted change;
5. export only after human review.

Only one agent request can run at a time.

## Run traces

Every run is written as JSON Lines to `~/.koharu/traces/agent/<run-id>.jsonl`, outside project storage and source control. The `started` event includes this absolute location, and `get_agent_trace_location(run)` returns it to the UI for a running or completed run.

Each line has `schema_version`, `run_id`, monotonically increasing `sequence`, `timestamp_unix_ms`, `event`, and `data`. Schema version 1 records:

- `run_started`: the exact prompt, a compact project-context envelope with a complete content-addressed artifact reference, and the image-capture policy;
- `model_text_delta` and `model_reasoning_delta`: each streamed delta;
- `tool_started`: call ID, tool name, parsed arguments, and the exact arguments JSON string;
- `tool_finished`: a bounded result envelope or error, `changed`, and image provenance;
- `payload_compacted`: payload kind, original/final serialized byte counts, and the path/media type/byte length/BLAKE3 reference for the complete payload;
- host-specific structured records between tool boundaries; the headless E2E host emits `source_analysis`, `sfx_skip_classification`, `ui_panel_anchor_verification`, `pipeline_telemetry`, `pipeline_preprocessing`, `acceptance`, `visual_review`, `agent_visual_semantic_review`, `source_evidence_inspection`, `page_source_debug_view`, `page_translation_review`, `page_debug_view`, `page_translation_revision`, `agent_correction`, and `export_telemetry` records, each with its own `schema_version`. SFX records retain the exact source identity, evidence binding, classifier, confidence, rationale, and explicit absence of target ownership. UI-panel records retain the required UI classification, exact source identity/crop/debug binding, finite panel and safe-interior geometry, original-raster border/contour/interior measurements, detector/version, rejection state, containment/intersection measurements, text-safe evidence, classifier, confidence, and association reason. Page-context and agent-review records include dossier/artifact digests and page revision provenance, never raw image bytes. Page semantic revision and in-loop review records bind all four supplied BLAKE3 digests to one exact page and scene revision. Preprocessing records every justified source-region merge and generated typography adjustment. Corrections preserve the original OCR plus the inspected or rejected scene revision, evidence source, rationale, and full before/after state. Acceptance includes configured thresholds, skipped-source evidence, original source bounds, the authoritative target layout-anchor kind/ID/bounds, source and target writing modes, association confidence/reason, per-anchor coverage/overflow, rasterized glyph-ink clearance and violations against the inset detected balloon, UI panel, or text-safe contour, pairwise overlap measurements, and every rejection. Geometry acceptance does not assert semantic fidelity: export also requires either the configured command judge or accepted in-loop structured review over the generated original/preview/semantic bundle. The bundle records the deterministic result and is retained for inspection even when those gates reject the run;
- `run_completed`, `run_failed`, or `run_cancelled`: final message or error and a compact final project-context envelope with its complete artifact reference.

Rendered evidence bytes are sent to Codex for the immediate inspection turn but are released from retained conversation afterward; artifact path and BLAKE3 provenance remain. Traces never embed image bytes or oversized raw tool results. Authentication headers, tokens, provider credentials, raw Codex response envelopes, and encrypted reasoning content are never traced. Prompts, compact project text, model deltas, and bounded tool arguments are trace data, so do not put credentials into an agent request.
