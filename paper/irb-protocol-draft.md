# Draft: IRB exemption request — protocol description

For the faculty PI to review/edit and submit via Cornell's RASS-IRB system.
Cornell-specific fields (PI info, department, funding) left blank.

---

**Title:** Human ratings of perceptual music similarity along melodic, rhythmic, and timbral dimensions

**Exemption category requested:** Exempt Category 3 — benign behavioral
intervention with adults, with information recorded such that subjects cannot
be identified.

**Purpose.** Machine-learning models of music similarity are typically
evaluated against proxy labels (genre tags) or synthetic data. This study
collects human perceptual ratings to serve as ground truth for a public
research benchmark: participants rate how similar pairs of short music
excerpts sound along three musical dimensions.

**Participants.** Adults (18+), recruited via university mailing lists,
word of mouth, and optionally the Prolific platform (compensated at or above
Prolific's fair-pay guideline). Target N ≈ 25–40. No vulnerable populations;
no minors. Hearing-impaired individuals may participate; normal hearing is
not required for validity since ratings are aggregated.

**Procedure.** Participants open an anonymous web survey. Each trial plays
two 30-second music excerpts (instrumental multitrack recordings licensed
for research use, from the MoisesDB and MUSDB18-HQ research corpora, plus
controlled manipulations of them). Participants rate similarity on three
0–100 sliders labeled *melody*, *rhythm*, and *timbre/instrumentation*.
A session lasts 15–25 minutes (~30–50 trials) and can be abandoned at any
time; partial data are kept only if the participant submits.

**Data collected.** Slider ratings, trial order/timing, an optional
self-reported musical-experience level (years of training, categorical),
and a randomly generated session identifier. No names, emails, IP addresses,
or demographic identifiers are recorded. Prolific IDs, if used, are stored
only for payment processing and deleted after payment, never linked to
published data.

**Risks and benefits.** Minimal risk — comparable to everyday music
listening. Participants are advised to set a comfortable volume. No direct
benefit to participants; societal benefit is a public, human-validated
evaluation resource for music-understanding research.

**Consent.** An information page precedes the survey (purpose, duration,
anonymity, voluntariness, contact info for the PI and Cornell IRB);
proceeding constitutes consent. No signature collected, preserving
anonymity.

**Data sharing.** Aggregated, anonymized ratings will be released publicly
as part of the benchmark (ratings keyed to audio-clip identifiers and
anonymous session IDs only).
