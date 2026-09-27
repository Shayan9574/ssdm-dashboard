# Framework Traceability Checklist

Maps every section of the methods document (and every box of
ssdm_workflow_v2.svg) to its implementation. A section is checked only
when its code is merged, verified, and visible in the dashboard.
Statuses: DONE, PARTIAL (with what remains), PENDING (checkpoint letter).

| Methods section | Workflow box | Implementation | Status |
|---|---|---|---|
| 1 Overview, four target pathogens | Start / scope | modules/agent_engine.py CORE4 | DONE |
| 2 Evidence base assembly | Assemble evidence base | data/curated_seed.xlsx + scripts/refresh_data.py | DONE |
| 3 Data regime typology (Types 1, 2, 3) | Classify by data regime | Data Dictionary sheet + modules/load_data.load_data_dictionary | DONE |
| 4 Agents O, H, L; recursion; Agent M merge | Route to acquisition agents; Agent M | Agent O and L: refresh_data.py + agent_m_data.py; recursion: launcher staleness loop. Agent H reconciliation (live level, historical dynamics): PARTIAL, live values sit beside baselines | PARTIAL (checkpoint C) |
| 5 Eight domain agents | Distribute to eight agents | modules/agent_engine.AGENTS + modules/agent_view.py, all eight pages on the shared renderer | DONE |
| 5 AI enrichment (additive only) | AI evidence enrichment | modules/genai.py + modules/evidence.py exist; wiring | PENDING (C) |
| 6 Surveillance pipeline, six feeds, weekly | Live sources | scripts/refresh_data.py + GitHub or launch refresh | DONE |
| 7.1 Fuzzy chain, GMIR, five term scale | Weight Manager | modules/weight_manager.py gmir, IMPORTANCE_SCALE | DONE |
| 7.2 Criterion level and agent level modes; multi expert mean | Weight Manager modes | criterion level DONE per agent; agent level store present, elicitation page | PENDING (D, feeds Section 16) |
| 7.3 Four input modes with provenance | Input modes | weight_manager.render_weight_elicitation (default, direct, upload with template, questionnaire with visible fuzzy chain) | DONE |
| 8 Readiness gate, refine and re run | Readiness diamond | agent_view readiness note + co-equal tier prompt; full gate across registry and scenarios | PARTIAL (C) |
| 9.1 to 9.8 Updated MOSDM | Run updated MOSDM | modules/mosdm_core.py, verified on Agent 1; guard_min_shared = 1 per decision | DONE |
| 9.4 SECOND GENERATION: graded attainment, two regime value function, discrimination calibrated anchors theta_j in [0.3, 0.7]; limits hierarchy expert > historical > pool median | Run updated MOSDM | mosdm_core attainment_mode (default graded_calibrated); historical_limits plumbed, populated at checkpoint C | DONE (engine); REMINDER: rewrite methods document Sections 9.3 to 9.5 to the second generation formulation, approved in chat, before submission |
| 9.3 Expert limits override medians | Acceptable limits | weight_manager.render_limits_editor + engine expert_limits | DONE |
| 10 Scenario construction from 2016 to 2026 history | Construct scenario set | Scenario Registry curated sheet: nine weighted scenarios (S1 to S9, S4 redefined to sustained Rt above one, S6 provisional) plus the ST1 economic stress test; four converging sources applied; per marker n_s and Y_s recorded | DONE (S6 marker window extension via full RESP-NET pull remains, checkpoint C2) |
| 11 Probability engine pi_s = f^alpha s^beta h^gamma; fuzzy intensity; lambda = 0.7; baseline outside | Estimate probability | modules/scenario_engine.probability_table (f from registry, s and h expert plus Gemini blend, exponents configurable, normalization over weighted scenarios only) + Scenarios & Probability page audit table | DONE |
| 12 Scenario wise re runs; AI tier interpretation | Re-run MOSDM per scenario | scenario_engine.run_scenario and tier_matrix (emphasis default profiles, elicited overrides); scenario selector on every agent page; AI tier interpretation | PARTIAL: interpretation module lands with the synthesis at checkpoint D |
| 13 Stability classification, displacement, trigger, exposure | Stability diamond | scenario_engine.stability_table: robust versus scenario sensitive, displacement with direction, triggers, exposure sum p_s, instability index sum p_s abs delta; stress test excluded | DONE |
| 14 Indicators R, Phi, sigma, SI | Probability weighted SSDM | | PENDING (D) |
| 15 Quadrant classes | Classify by priority and stability | | PENDING (D) |
| 16 Second order matrix, agent weights W_g | Final integration | Agent 5 direction reconciliation noted in agent_engine docstring | PENDING (D) |
| 17 Cross agent SSDM | Apply SSDM across agents | | PENDING (D) |
| 18 Final synthesis (AI, never alters results) | Generate final synthesis | | PENDING (D) |
| 19 Outputs page | Outputs | | PENDING (D/E) |
| Standing rule: calculation audit beside every derived number | | Derivations sheet + load_derivations + agent_view expander | DONE |
| Standing rule: engine provenance label on results | | agent_view ENGINE_BADGE | DONE |
| Standing rule: co-equal tier asks the user in the dashboard | | agent_view warning prompt with elicitation path | DONE |
| Full interface rebuild (speed, design) | | | PENDING (E) |
