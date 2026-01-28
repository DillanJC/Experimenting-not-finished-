# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.1] - 2026-01-28

### Added
- **Paper draft**: `paper/PAPER_DRAFT.md` — full manuscript ready for submission
- **Detection rule demo**: `examples/detection_rule_demo.py` — practical usage example
- **Key findings analysis**: `experiments/key_findings_analysis.py` — discovery validation

### Changed
- **README.md**: Complete rewrite leading with key discovery (topology > flow)
- Documentation now emphasizes the "narrow passage" interpretation

### Key Discovery Documented
- `participation_ratio` (r=-0.529 on borderline) is strongest predictor
- `spectral_entropy` (r=-0.496 on borderline) is second strongest
- Negative correlation: LOW values = HIGH risk (constrained geometry)
- Simple detection rule: 48.6% precision at 32.7% coverage

## [1.5.0] - 2026-01-27

### Added
- **Phase 3 Trajectory Features**: For generative models with embedding logging
  - `drift_mean`, `drift_p95`: Speed through embedding space
  - `curvature_median`: Jerkiness / sharp turns
  - `smoothness`: Inverse jerkiness (stability indicator)
  - `total_distance`: Path length through space
  - `attractor_distance`: Distance to density modes
  - `TrajectoryLogger` helper class for logging during generation
  - `compute_settling_behavior()` for multi-run dispersion analysis
- All 5 phases now implemented (0, 1, 2, 3, 4, 5)

## [1.4.0] - 2026-01-27

### Added
- **Phase 1 Flow Features**: `local_gradient_magnitude`, `gradient_direction_consistency`, `pressure_differential`
  - Mean-shift based density gradient estimation
  - Adaptive bandwidth with per-point scaling
  - +8.68% improvement on borderline cases
- **Phase 2 Weather Features**: `turbulence_index`, `thermal_gradient`, `vorticity`
  - Atmospheric metaphor for embedding dynamics
  - Boundary-focused gradient measurement
- **Phase 4 State Mapping**: 6 interpretable cognitive states
  - `coherent`, `confident`, `constraint_pressure`, `novel_territory`, `searching`, `uncertain`
  - Percentile-based thresholds with soft assignment
- **Phase 5 Topology-Lite**: `d_eff`, `spectral_entropy`, `participation_ratio`
  - SVD-based local dimensionality estimation
  - `participation_ratio` shows r=-0.529 on borderline (strongest signal!)
- **Unified Pipeline**: `compute_safety_diagnostics()` single entry point
  - Returns `SafetyDiagnostics` with all 14 features + state mapping
  - `get_high_risk_mask()` for flagging uncertain/novel samples
- **Evaluation Scripts**: `phase1_flow_evaluation.py`, `phase2_5_evaluation.py`, `full_pipeline_demo.py`

### Changed
- Version bumped to 1.4.0
- Module exports expanded in `__init__.py`

### Technical Details
- Phase 5 topology features outperform Phase 1 on borderline cases
- Full pipeline provides +3.10% R^2 over Tier-0 baseline
- GPU acceleration available via PyTorch (optional)

## [1.0.0] - 2026-01-22

### Added
- **Core API**: `GeometryBundle` class for computing 7 geometric features
- **Advanced Features**: S-score, class-conditional Mahalanobis distance, conformal prediction
- **Performance**: Optional FAISS backend for scalable nearest neighbor search
- **Evaluation**: Comprehensive harness testing features on synthetic datasets
- **Documentation**: Complete API docs, examples, and tutorials
- **Testing**: Unit tests with 85%+ coverage and CI pipeline
- **Benchmarks**: Performance suite with scaling analysis

### Changed
- API simplified to dict-based feature returns
- Enhanced validation with reproducible evaluation templates

### Technical Details
- Rigorous evaluation identifies `knn_std_distance` as top uncertainty signal
- Boundary-stratified analysis validates improvements in high-uncertainty regions
- Compatible with Python 3.9+

## [0.1.0] - 2026-01-22

### Added
- Initial implementation of geometric safety features
- Basic evaluation on synthetic datasets
- Core functionality for AI safety diagnostics