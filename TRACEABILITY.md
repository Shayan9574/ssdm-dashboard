# Traceability: method (thesis Chapter 2, paper Section 4) to code

Status at Checkpoint H. Every row is covered by the verification suite in
`tests/`.

| Method element | Equation or section | Implementation |
|---|---|---|
| Orientation, coverage (observed for at least two alternatives), no imputation | Notation, Orientation, and Coverage | `mosdm_core.run_mosdm` (coverage, unstratified alternatives), `live_subcriteria.active_criteria` |
| Fuzzy evidence, GMIR | Eq. ch2_gmir | `utils_numeric.parse_fuzzy`, `parse_numeric` (ranges and bounds as interval trapezoids) |
| Limits hierarchy: expert, historical quantile, pool median (frozen below two observations) | Eq. ch2_limits | `run_mosdm`; history window in `live_subcriteria.window_start` (since 11 May 2023, then five rolling seasons) |
| Graded attainment, limit position over the pool together with the limit | Eq. ch2_normalize, ch2_attainment | `mosdm_core._orient`, `_graded_degree` |
| Anchor modes (fixed, calibrated with ties to the midpoint of the bounds, binary) | Eq. ch2_calibrated | `mosdm_core._calibrate_theta`; settings `theta_fixed`, `theta_min`, `theta_max` |
| Weighted attainment | Eq. ch2_weighted | `run_mosdm` |
| Tolerance tiers | Eq. ch2_tier | `run_mosdm` (strict, 1e-9 guard) |
| Dominance guard on informative shared criteria, q_min | Eq. ch2_guard | `mosdm_core._strictly_dominated`; settings `dominance_guard`, `guard_min_shared` |
| Target, within tier distance (crisp or discretized fuzzy), mid ranks for ties | Eq. ch2_target, ch2_score, ch2_discretize | `run_mosdm`, `MOSDMResult.ranks` |
| Attainment margin | Eq. ch2_margin | `MOSDMResult.separations` |
| Audit trail and explanations | Complete Procedure | `IterationRecord`, `MOSDMResult.explanations`; Method Trace page |
| Scenario as weight profile and limit overrides | Eq. ch2_emphasis (data derived multipliers) | `scenario_engine.scenario_weights`, `emphasis_multipliers`, `data/scenario_emphasis.json` |
| Admission from documented history, episodes, observable window | Admission from Documented History | `scenario_admission` (weekly markers S1 to S7), `scenario_engine.admission_table` |
| Tightening at the admission quantile; Rt at 1 | Scenario limit overrides | `scenario_engine.scenario_limit_overrides` |
| Scenario weights | Eq. ch2_scen_weight, ch2_frequency | `scenario_engine.probability_table` (full precision) |
| Severity and impact, respondents, assessor blend | Eq. ch2_blend | `scenario_engine._blend`, `set_expert_terms`, `set_ai_assessment` |
| Re stratification, displacement, exposure, instability | Eq. ch2_displacement to ch2_instability | `scenario_engine.run_scenario`, `stability_table`, `tier_matrix` |
| Indicators R, Phi, sigma, SI | Eq. ch2_expected_rank to ch2_si | `ssdm_core.indicators` |
| Typology | Table ch2_typology | `ssdm_core.quadrants` |
| Second order matrix, reversal, parallel baseline | Eq. ch2_reverse | `ssdm_core.second_order` |
| Second order stratification with W_g | Second Order Stratification | `ssdm_core.cross_agent_run` |
| Linguistic chain, scales | Table ch2_scales | `weight_manager.IMPORTANCE_SCALE`, `INTENSITY_SCALE`, `gmir` |
| Aggregation rules (mean; minimum, geometric means, maximum) | Eq. ch2_aggregate_base | `weight_manager.aggregate_trapezoids`; setting `aggregation_rule` |
| Input modes and provenance | Input Modes and Provenance | `weight_manager.render_weight_elicitation`, `state` (author and time on every input) |
| Readiness gate | paper Section 4.4 | `readiness.evaluate` |
| Sensitivity in the app | Evaluation Protocol (subset) | `sensitivity.py`; the full protocol runs from the experiment scripts |
| External evidence as information | paper Section 5.4 | `evidence_gate` (never enters the engine) |
| Weekly data, replay, reproducibility | Temporal Replay | `.github/workflows/weekly-data.yml`, `scripts/refresh_data.py`, `scripts/export_run.py --as-of` |
