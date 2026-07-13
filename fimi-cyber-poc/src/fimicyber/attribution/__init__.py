"""Public-source actor-attribution support.

This package ranks reported actor hypotheses for analyst review. It does not
make legal findings or identify natural persons.
"""

from fimicyber.attribution.hypotheses import (
    build_attribution_graph,
    build_attribution_hypotheses,
    evaluate_attribution,
)
from fimicyber.attribution.calibration import (
    build_error_analysis,
    calibrate_hypotheses,
    evaluate_attribution_scopes,
)
from fimicyber.attribution.external_case import run_external_ghostwriter_case
from fimicyber.attribution.generalization import run_multiactor_generalization
from fimicyber.attribution.provenance import build_evidence_provenance

__all__ = [
    "build_attribution_graph",
    "build_attribution_hypotheses",
    "evaluate_attribution",
    "build_error_analysis",
    "calibrate_hypotheses",
    "evaluate_attribution_scopes",
    "run_external_ghostwriter_case",
    "run_multiactor_generalization",
    "build_evidence_provenance",
]
