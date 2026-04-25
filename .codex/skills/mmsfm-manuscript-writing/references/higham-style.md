# Higham Note: Readability Heuristics

This file distills the writing habits that have been most useful in this manuscript from Nicholas Higham's SIAM News note on style together with recurring MMSFM manuscript failure modes across the whole methodology.

Use this file as a heuristic aid, not as a prescriptive style guide. Higham's note is an essay about style awareness, readability, and cliche avoidance. It is not a formal set of SIAM writing rules.

## Core Standard

Write for a mathematically literate reader who wants the point quickly and precisely. Favor direct scholarly prose over rhetorical setup.

## Practical Rules

- Start each paragraph with its controlling claim, not with scene-setting.
- Keep subsection titles descriptive and aligned with what the subsection actually does.
- Avoid cliche or filler phrases such as `the key point is`, `it is worth noting`, `what matters here`, or `in this sense` when a direct statement will do.
- Avoid defensive phrasing such as `we do not claim`, `should be read only as`, `not intended to`, or `does not ... but ...` unless the distinction is mathematically essential.
- Avoid table-of-contents prose in the main argument. Replace `Appendix X develops...` with the mathematical consequence needed here, and leave the full derivation to the appendix.
- Introduce notation and concepts close to their first actual use.
- Keep claims local. Do not advertise posterior, MMSE, Fisher--Rao, or spacetime structures in a paragraph whose job is only to explain a practical role.
- Prefer one sharp sentence to two vague synonyms.
- End paragraphs on the mathematical point, not on a generic summary.
- Keep the sentence honest about its job: definition, claim, proof consequence, implementation statement, experiment setup, or interpretation.
- Prefer natural prose over artificially compressed prose. A paragraph can be concise without sounding mechanical.

## MMSFM-Specific Checks

- If the paragraph is about the method, say what stage it belongs to:
  `representation learning`, `bridge or downstream transport`, `evaluation`, or `discussion of scope`.
- If the paragraph is about training, make sure the prose matches the actual objective already defined in the manuscript.
- If the paragraph is about geometry, say which geometry:
  `decoder chart geometry`, `observation geometry`, `spacetime geometry`, or `posterior/Fisher--Rao geometry`.
- If the paragraph is about a theorem or corollary, separate the proved statement from the practical implication.
- If the paragraph is about experiments, replace procedural detail with the claim being tested and the conclusion supported by the reported result.
- Keep appendix-only structures in the appendix unless the reader needs them immediately to understand the claim in the main text.
