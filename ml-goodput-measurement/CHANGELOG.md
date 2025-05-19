# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allows:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [0.0.1] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[0.0.1]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->
## [0.0.10] - 2025-04-28

* Support for custom badput events which are synchronous and training-overlapped.
* Handling of edge case caching scenario.

## [0.0.9] - SKIPPED

* Used for external testing. Please upgrade to 0.0.10.

## [0.0.8] - 2025-04-03

* Fix computation of ideal step time when step_times is empty.

## [0.0.7] - 2025-03-24

* Cache updates to Other/Unknown Badput.
* Exclude monitoring asynchronous Badput types in GCM.
* Total and last step updates with hidden events.
* Interval Query Monitoring in GCM.

## [0.0.6] - 2025-03-17

* Updates to data loading Badput buckets (Separated into Async & Sync).
* Short term fix to Pathways SuspendResume anomalous step time detection.
* Updates to account for Pathways Elastic Training.
* Automatic asynchronous upload of goodput, badput and step time deviation metrics to GCM.

## [0.0.5] - 2025-02-03

* Goodput Cache and library improvements.
* Query and Monitor API support for checkpoint save and restore.
* Interval Query API support.
* Query and Monitor API support for step time deviation.

## [0.0.4] - 2024-09-13

* Add Badput breakdown to GoodputMonitor.
* Add Checkpoint Badput Calculator backend.
* Return last recorded step from Goodput query API.
* Bug Fixes
  * Fix a potential race-condition with Tensorboard write to GCS.
  * Fix zero job time issue on long running jobs

## [0.0.3] - 2024-05-28

* Compute and discount Badput from first step after start or restart.
* Compute and discount Badput due to anomalous step times (Pathways only).
* Badput recording APIs
* Some Badput computation APIs (TPU initialization , training preparation, data loading, program startup)
* Goodput monitoring API to asynchronously query and upload Goodput to Tensorboard.
* Bug Fixes
  * Fix Goodput calculation with disruptions
  * Fix some Cloud Logging latency and batching issues.

## [0.0.2] - 2024-02-29

* Bug Fixes
  * Fixes a typing mismatch in total step time calculation.
* Code and documentation cleanup

## [0.0.1] - 2024-02-26

* Initial release of ML Goodput Measurement PyPi package
* Feature: Contains the Goodput module which allows logging and retrieval of training job's overall productive Goodput

[0.0.10]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.8...v0.0.10
[0.0.8]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/AI-Hypercomputer/ml-goodput-measurement/releases/tag/v0.0.1