# Third-party notices

This research repository contains material from the sources below. Their terms
apply to that material, including excerpts retained in historical experiment
records. This document does not grant a license to the project-specific code.

## Allen fitted cells and model rules

Allen Institute for Brain Science, **Allen Cell Types Database**, models
503612028 and 523000114. [Pinned provenance](data/allen/README.md) identifies
the downloaded configurations, metadata, source URLs and checksums.

The [Allen Institute Terms of Use](https://alleninstitute.org/terms-of-use/) and
[Citation Policy](https://alleninstitute.org/legal/citation-policy) apply to
the dataset. It is retained here for noncommercial research. The configured
model equations follow [AllenSDK](https://github.com/AllenInstitute/AllenSDK);
[its license is reproduced here](licenses/AllenSDK-LICENSE.txt). AllenSDK's
license is not an unrestricted BSD license: it additionally restricts
commercial redistribution.

## Vendored Rust dependencies

`src/humanbrain/_native_src/vendor/` contains the locked native dependencies.
Each package declares `MIT OR Apache-2.0` and carries its own copyright and
license files. Keep those package notices with copies of the vendored code.
`Cargo.lock` records versions; these licenses cover the dependencies, not the
rest of this repository.

## Software-task research transcripts

The `artifacts/swe-dev-*` and `logs/run_evaluation/` records include attempts at
the public `sqlfluff__sqlfluff-1625` task, including source excerpts, patches,
commands and outputs. They are retained as experiment evidence, not as private
conversations or claims of a general-purpose coding capability. Local model
endpoints and dummy API-key values in the original records are run configuration,
not production service credentials.

- [SQLFluff](https://github.com/sqlfluff/sqlfluff), copyright Alan Cruickshank:
  [MIT license](licenses/SQLFluff-MIT.txt).
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), copyright
  Kilian A. Lieret and Carlos E. Jimenez:
  [MIT license](licenses/mini-swe-agent-MIT.txt).

## Reading materials

The small Crown-copyright reading-test extract and route map carry their
[Open Government Licence attribution and source-custody notes](experiments/reading_exam_development/ks2_2026_amelia/README.md).
Other experimental corpora and optional model downloads retain their own terms;
source references and hashes in a result record do not transfer rights to those
external datasets or models.
