# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

- Consolidated installation, configuration, API, diagnostics, and troubleshooting guidance
  in the README and removed the overlapping HOWTO document.
- Updated tutorial notebooks to use the public acceptance-rate diagnostic.

### Fixed

- Source provenance now prefers the revision recorded by a VCS installation and no longer
  reports an unrelated repository surrounding the active environment.
- Reusing an output directory removes stale optional plots that the new run disables.
- CLI configurations with fewer than two samples are rejected before sampling, and
  diagnostic failures are reported as command-line errors instead of tracebacks.
- Acceptance-rate calculations exclude the initial chain state, which is not a proposal.
- The Gaussian example target rejects non-scalar, non-finite, or non-positive standard
  deviations.

## [0.2.0] - 2026-08-12

### Added

- Reusable `RandomWalkMetropolisHastings` class alongside the function API.
- Input validation for sampler arguments and command-line configuration.
- Reproducibility metadata containing requested and effective configuration, the resolved
  random seed and bit generator, source provenance, environment details, summary statistics,
  and SHA-256 fingerprints of output arrays.
- Tests for target distributions, invalid inputs, metadata replay, output fingerprints,
  dashboard generation, and minimal runtime installation.
- Ruff formatting and linting checks in continuous integration CI.

### Changed

- Replaced the previous weakly curved example target with the two-dimensional Haario banana
  distribution and updated its examples, proposal scales, tests, and tutorials (proposed in Final Presentation for this class).
- Moved SciPy out of the runtime dependencies and into the development and notebook.
- Expanded and synchronized the README, HOWTO, example configuration, and API documentation.
- Standardized package, citation, and license author metadata.

### Fixed

- Unknown or misspelled configuration keys are now rejected instead of silently falling back
  to defaults.
- Invalid configurations fail before sampling starts or output files are created.
- The minimal-install CI smoke test now uses the tuned Haario banana configuration.

## [0.1.1] - 2026-08-10

### Fixed

- Added CI testing on Python 3.10 and 3.13.
- Restored public API coverage and removed references to unimplemented functionality.
- Corrected the MIT license classifier and related documentation.

## [0.1.0] - 2026-07-20

### Added

- Initial release with Random-Walk Metropolis-Hastings sampling, diagnostics,
  visualization, example targets, and a reproducible command-line interface.

[Unreleased]: https://github.com/korkmazgoz-caglar/random-walk-mcmc/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/korkmazgoz-caglar/random-walk-mcmc/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/korkmazgoz-caglar/random-walk-mcmc/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/korkmazgoz-caglar/random-walk-mcmc/releases/tag/v0.1.0
