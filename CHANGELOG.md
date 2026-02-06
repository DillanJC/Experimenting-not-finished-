# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-01-28

### Added
- **Equalizer Test: Applied Geometric Telemetry**
  - `experiments/equalizer_test/reasoning_trace_logger.py` — reasoning trace with geometric monitoring
  - `experiments/equalizer_test/build_reference_corpus.py` — diverse reference corpus builder
  - `experiments/equalizer_test/visualize_trace.py` — trajectory visualization
  - `experiments/equalizer_test/EQUALIZER_TEST_REPORT.md` — full experiment report

### Key Results (Equalizer Test)
- **PR tracks cognitive expansion**: TROUGH (8.01) at constraint, PEAK (9.82) at insight
- **Insight pattern detected**: TROUGH -> EXPANSION -> DROP -> PEAK
- **Novelty score**: 0.250 (moderate) for novel audio filter design
- **11 reasoning steps parsed** from Claude's response to equalizer problem

## [1.8.0] - 2026-01-28

### Added
- **Track 3: Meta-Cognitive / Geometric State API**
  - `mirrorfield/api/state_monitor.py` — `GeometricStateMonitor` class
  - `experiments/track3_api/live_detection_demo.py` — real-time agent simulation
  - `experiments/track3_api/parallax_integration.py` — multi-AI pipeline integration
  - `experiments/track3_api/TRACK3_REPORT.md` — full API documentation

### Key Features (Track 3)
- **Real-time self-monitoring**: `monitor.get_state(embedding)` returns state, flags, features
- **Batch analysis**: G ratio and CV differentiate clean (0.50, 0.31) from poison (0.95, 0.03)
- **Modulation advice**: Actionable suggestions for agent behavior modification
- **Integration patterns**: LangChain callbacks, LlamaIndex handlers, Parallax wrapper

## [1.7.0] - 2026-01-28

### Added
- **Track 2: Architectural Insight & Physics Analogies**
  - `experiments/track2_physics/lambda_run.py` — G ratio uniformity analysis
  - `experiments/track2_physics/decoherence_chamber.py` — multi-scale entropy evolution
  - `experiments/track2_physics/TRACK2_REPORT.md` — full experimental report

### Key Results (Track 2)
- **Lambda Run**: Cluster-poisoned samples show G = 0.95 (vs 0.50 for clean) — uniformly constrained geometry
- **Coefficient of Variation**: CV = 0.033 for poison vs 0.326 for clean — 10x less variation
- **Decoherence Chamber**: Poisoned samples have lower entropy at ALL 7 depth levels (8-256 dims)
- **Physics analogy validated**: Poisoning induces "condensate-like" uniform geometry

## [1.6.0] - 2026-01-28

### Added
- **Track 1: Poison Detection Experiment** — validates geometric features for backdoor detection
  - `experiments/track1_poison/create_poisoned_dataset.py` — label-flip backdoor injection
  - `experiments/track1_poison/train_classifiers.py` — clean vs poisoned model training
  - `experiments/track1_poison/compute_geometry.py` — geometric feature analysis
  - `experiments/track1_poison/detect_poison.py` — detection rule with AUC evaluation
  - `experiments/track1_poison/TRACK1_REPORT.md` — full experimental report

### Key Results (Track 1)
- **Cluster poisoning**: ROC-AUC = 0.947 using `participation_ratio` alone
- **Boundary poisoning**: ROC-AUC = 0.785 using `d_eff`
- **Topology features detect poisoning**: Same features that predict behavioral instability also identify poisoned samples
- **Constrained geometry = poison signature**: Poisoned samples show LOWER participation_ratio, spectral_entropy, and d_eff

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