"""Value-aware block utility and budgeted submodular selection."""

from mosaickv.selection.calibration import (
    AdaptiveHeadRouting,
    CalibrationStabilityError,
    FidelityCalibrationProfile,
    FidelityObservation,
    HeadFidelityObservation,
    derive_adaptive_head_routing,
    select_fidelity_profile,
)
from mosaickv.selection.exhaustive import compare_greedy_to_exhaustive, exhaustive_select
from mosaickv.selection.objective import BudgetedObjective, audit_objective_properties
from mosaickv.selection.selector import (
    InfeasibleSelectionBudgetError,
    lazy_greedy_select,
    select_all_exact,
)
from mosaickv.selection.types import (
    BlockUtility,
    BlockUtilityTable,
    ExhaustiveSelectionResult,
    GreedyOptimalityComparison,
    ObjectiveBreakdown,
    ObjectivePropertyReport,
    SelectionBudget,
    SelectionDecision,
    SelectionReason,
    SelectionResult,
)
from mosaickv.selection.utility import HeadId, compute_block_utilities

__all__ = [
    "AdaptiveHeadRouting",
    "BlockUtility",
    "BlockUtilityTable",
    "BudgetedObjective",
    "CalibrationStabilityError",
    "ExhaustiveSelectionResult",
    "FidelityCalibrationProfile",
    "FidelityObservation",
    "GreedyOptimalityComparison",
    "HeadFidelityObservation",
    "HeadId",
    "InfeasibleSelectionBudgetError",
    "ObjectiveBreakdown",
    "ObjectivePropertyReport",
    "SelectionBudget",
    "SelectionDecision",
    "SelectionReason",
    "SelectionResult",
    "audit_objective_properties",
    "compare_greedy_to_exhaustive",
    "compute_block_utilities",
    "derive_adaptive_head_routing",
    "exhaustive_select",
    "lazy_greedy_select",
    "select_all_exact",
    "select_fidelity_profile",
]
