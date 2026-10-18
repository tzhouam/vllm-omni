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
