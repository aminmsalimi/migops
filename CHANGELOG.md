# Changelog

## Unreleased

### Added

- GPU and MIG status inspection
- MIG profile discovery
- workload discovery
- environment diagnostics
- Smart Split recommendations
- simple GI + CI creation
- safe CI-before-GI destruction
- MIG mode control
- advanced GI and CI management
- YAML desired-state configuration
- validate, diff, and plan workflows
- snapshots and restore
- safe apply with pre-change snapshot and final verification
- JSON output for automation-oriented commands
- GitHub Actions tests

### Safety

- workload checks before disruptive apply operations
- dry-run support
- explicit `--yes` requirement for real desired-state apply
