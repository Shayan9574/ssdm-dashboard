# Checkpoint H

## Install over your local folder

1. Unzip over `ssdm-dashboard-skeleton` (keep your `.git` folder).
2. Delete the files this checkpoint removes (they are no longer used):

```bash
git rm diagnose.py modules/status_banner.py modules/evidence.py modules/scenarios.py \
       modules/agents/agent1.py modules/agents/agent2.py modules/agents/agent3.py \
       modules/agents/agent4.py modules/agents/agent5.py modules/agents/agent6.py \
       modules/agents/agent7.py modules/agents/agent8.py \
       scripts/export_inputs.py scripts/export_results.py
git add -A
git commit -m "Checkpoint H: web deployment, weekly data, method fixes, verification suite"
git push
```

3. Follow "Deploying (one time)" in README.md.

## What changed

Method (engine)
* Limit position measured over the pool together with the limit: an
  alternative below an external limit no longer receives the credit of
  reaching it.
* Pool median frozen once fewer than two pool members are observed.
* An alternative observed on no weighted active criterion is reported as
  not stratifiable (never placed at the bottom); its cell of the second
  order matrix is missing.
* Constant criteria excluded from the guard, the target and the distances.
* Ties within a tier share the average rank (mid ranks); target distance 0.
* Calibrated anchor ties resolve to the midpoint of the bounds.
* Tie breaking robust to floating point noise.
* Fuzzy distance option (discretized fuzzy distance on ranged evidence).
* Ranged evidence parsed as interval fuzzy numbers ("<1%" is [0, 1]).

Scenario layer
* Episodes recomputed every week from the CDC series for S1 to S7 (gap
  tolerance three weeks, minimum two weeks; S4 three consecutive weeks);
  curated counts shown beside them; S8 and S9 from the event catalog.
* Emphasis k = 3 replaced by data derived multipliers (episode median over
  window median, never below one); explicit emphasized criteria in
  `data/scenario_emphasis.json`; expert scenario weights override.
* Tightening at P90 of the history (the admission quantile), Rt at 1; a
  scenario never loosens an expert limit.
* History window: since 11 May 2023, then rolling five seasons.
* Scenarios without an episode are weighted zero; replay uses only episodes
  dated on or before the replay date.
* Full precision throughout; several expert respondents for severity and
  impact with either aggregation rule; Poisson interval on every f_s.

Data
* National Rt: the latest model run per date, seven day median; values older
  than four weeks are treated as missing (the national CFA series in the
  workbook ends 26 May 2026).
* NREVSS positivity is national only: missing in the Michigan scope.
* Wastewater labelled as Michigan sewersheds in every scope.
* NNDSS national series: US residents row (the label changed in 2025).
* Agent 7 unknown spread status stays missing; no zero filling anywhere.

Application
* Stale results fixed: every cached result is keyed on all inputs,
  settings and data; no result is shared between users with different
  inputs.
* Inputs saved to a Google Sheet (local file when run locally), with author
  and time; editor password; viewers explore read only.
* Disease selection respected in scenario runs; results update immediately
  after Apply; limits editor shows the real default and its source.
* Method Settings page (every option of the method), Method Trace page,
  run manifest download, update log of the weekly data.
* External evidence is information for decision makers: validated figures
  beside the dashboard's values, never rescaled, never changing a result.
* Weekly GitHub workflow: download, recompute, verify, publish to the data
  branch, archive as a release. Colab is optional.
* Verification suite (75 tests) in `tests/`.

## Effects to carry into Chapter 2 and the paper

* Eq. ch2_attainment text: limit position over the pool together with the
  limit (replaces clipping); frozen median; unstratified alternatives;
  noninformative criteria excluded from guard and distances; mid ranks.
* Eq. ch2_emphasis: data derived multipliers replace k = 3.
* Tightening at the admission quantile and the stricter of expert and
  scenario limits; history window; episode rule; replay truncation.
* A margin below the tolerance marks a boundary imposed by the guard.
* Every number of Sections 6 and 7 of the paper must be regenerated with
  this engine.
