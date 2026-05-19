from __future__ import annotations

from scripts.biological_models.growth_models import CurveFitGrowthModel, logistic, monod_proxy, _p0_proxy, _p0_sigmoid
from scripts.forecasting.base import ForecastModel, SkippedModel


def get_differential_equation_models() -> list[ForecastModel]:
    models: list[ForecastModel] = [
        CurveFitGrowthModel("ODE_Logistic", logistic, _p0_sigmoid, note="Analytical logistic ODE solution."),
        CurveFitGrowthModel("ODE_Monod_Chemostat", monod_proxy, _p0_proxy, note="Chemostat proxy without inflow/substrate sensors."),
        CurveFitGrowthModel("Chemostat_Model", monod_proxy, _p0_proxy, note="Chemostat proxy without dilution-rate data."),
        SkippedModel("PDE", "differential_equations", "PDE requires spatial grid or reactor field data."),
        SkippedModel("DDE", "differential_equations", "DDE requires explicit delay structure or delayed covariates."),
        SkippedModel("Reaction_Diffusion", "differential_equations", "Reaction-diffusion requires spatial concentration data."),
        SkippedModel("Population_Balance_Model", "differential_equations", "Population balance requires size distribution data."),
        SkippedModel("dFBA", "differential_equations", "dFBA requires a metabolic network model and exchange fluxes."),
    ]
    for model in models:
        model.category = "differential_equations"
    return models
