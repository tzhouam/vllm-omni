# Historical calibration evidence

The original `*_unrecorded.json` records omitted optimization switches or decoder
versions. They are retained byte for byte for historical method and rejection
tests. Do not add current provenance to old data.

`cuda_nightly_quality_population_within_cohort.json` is the recorded-zero
population before pooling cross-cohort reference comparisons. It retains its
original six within-cohort pixel summaries and six native reference scores per
case. The active version 2 record uses every unordered pair of the same six
reference videos. This archive remains unchanged for boundary and derivation
checks; archiving it does not change original hardware pass/fail results.

Archived records are not loaded by the CI runner.
The active nightly records use independent recorded-zero H100 reference cohorts;
independent final-source target-hardware validation is still required.

The production reference was advanced to `4134c7f421a195407da6f779f21c3e40c93a2440`
before fresh collection on 2026-10-19. The earlier records retained outside
`history/` are now incompatible with that pin and intentionally fail contract
validation. Their original data and failures remain unchanged; qualified fresh
replacements must archive those predecessors byte for byte. See the suite's
[current production protocol](../../README.md#current-production-reference-calibration-pending).

`profiles/` preserves all seven earlier profiles byte for byte from
`7d1dfdafbec699b8fb327bd5375a23281dfba2df`. Historical evidence hashes bind to
these snapshots after the active reference pins advance; the active profiles
differ only in `reference_revision`. Do not rewrite archived profile hashes to
make old observations appear to have used the new production reference.

The intermediate `1cc64011cee7ed4860f8a3849511130df747bc50` pin was superseded
before collecting any fresh hardware data: the actual GitHub merge included
newer accepted main changes. The final production snapshot includes upstream
`990d4c1061124bb87ba3b57a8c1095c2799b41aa`; the reference-only protocol and
candidate regression allowances are unchanged.
