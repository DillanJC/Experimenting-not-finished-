"""
Mirrorfield Geometry Module — Phase E

Native k-NN geometric features for AI safety boundary detection.

Validated Performance:
- Baseline (embeddings only): R² ≈ 0.34
- With geometry features: R² ≈ 0.40 (+6.4%, p<10⁻⁶)
- Robust across k ∈ {25, 50, 100}

Main Interface:
    from mirrorfield.geometry import GeometryBundle

    # Initialize with reference set
    bundle = GeometryBundle(reference_embeddings, k=50)

    # Compute for query points
    results = bundle.compute(query_embeddings)

    # Extract features for ML models
    features = bundle.get_feature_matrix(results)
"""

from .schema import (
    GeometryOutputSchema,
    SCHEMA_VERSION,
    validate_output,
    create_record,
    NAME_MAP
)

from .features import (
    compute_knn_features,
    compute_centroid_anchor,
    detect_dark_rivers,
    detect_observer_mode,
    batch_invariance_test,
    FEATURE_NAMES
)

from .bundle import GeometryBundle

# Phase 1 Flow Features (experimental)
from .phase1_flow_features import (
    compute_phase1_features,
    compute_phase1_features_fast,
    compute_phase1_features_gpu,
    compute_gradient_magnitude_only,
    combine_with_tier0_features,
    get_combined_feature_names,
    PHASE1_FEATURE_NAMES,
    PHASE1_FEATURE_NAMES_FAST,
    PHASE1_FEATURE_NAMES_RECOMMENDED,
    DEFAULT_K,
    DEFAULT_KC,
    DEFAULT_BANDWIDTH_MULTIPLIER,
)

# Phase 2 Weather + Phase 5 Topology-Lite Features
from .phase2_weather_features import (
    compute_phase2_features,
    compute_topology_lite_features,
    compute_turbulence_index,
    compute_thermal_gradient,
    PHASE2_FEATURE_NAMES,
    PHASE5_FEATURE_NAMES,
    PHASE2_5_FEATURE_NAMES,
)

# Phase 4 State Mapping
from .phase4_state_mapping import (
    compute_state_scores,
    get_state_summary,
    StateThresholds,
    STATE_NAMES,
    STATE_DESCRIPTIONS,
)

# Phase 3 Trajectory Features (requires trajectory data)
from .phase3_trajectory_features import (
    compute_phase3_features,
    compute_trajectory_stats,
    compute_settling_behavior,
    compute_attractor_proximity,
    TrajectoryLogger,
    PHASE3_FEATURE_NAMES,
)

# Unified Pipeline (main entry point)
from .unified_pipeline import (
    compute_safety_diagnostics,
    print_diagnostics_summary,
    SafetyDiagnostics,
)

__version__ = "1.5.0"  # All phases complete

__all__ = [
    # Main interface
    'GeometryBundle',

    # Schema
    'GeometryOutputSchema',
    'SCHEMA_VERSION',
    'validate_output',
    'create_record',
    'NAME_MAP',

    # Tier-0 Features
    'compute_knn_features',
    'compute_centroid_anchor',
    'detect_dark_rivers',
    'detect_observer_mode',
    'batch_invariance_test',
    'FEATURE_NAMES',

    # Phase 1 Flow Features (experimental)
    'compute_phase1_features',
    'compute_phase1_features_fast',
    'compute_phase1_features_gpu',
    'combine_with_tier0_features',
    'get_combined_feature_names',
    'PHASE1_FEATURE_NAMES',
    'PHASE1_FEATURE_NAMES_FAST',
    'DEFAULT_K',
    'DEFAULT_KC',
    'DEFAULT_BANDWIDTH_MULTIPLIER',
]
