# SSDM Public Health Decision Support Dashboard

A web application that stratifies infectious diseases for public health
prioritization with the extended stratification method (thesis Chapter 2):
eight perspectives (domain agents), a scenario layer built from the
documented history of the US health system, probability weighted stability
indicators, and a second order integration. Curated evidence is combined
with seven CDC surveillance feeds that refresh automatically every week.

## How it runs

| Part | Where | What it does |
|---|---|---|
| Web app | Streamlit Community Cloud, from this repository's `main` branch | Serves the dashboard to anyone with the link |
| Weekly data | GitHub Actions, `.github/workflows/weekly-data.yml` | Every Saturday downloads the CDC feeds, recomputes the derived quantities, writes them to the `data` branch, and archives the week as a release |
| Expert inputs | A Google Sheet (or a local file when run locally) | Weights, limits, respondents, severity and impact ratings, saved settings, external evidence ledger, with author and time |
| Curated evidence | `data/curated_seed.xlsx` in this repository | Baseline evidence, scenario registry, dictionary, citations, derivations |

Viewers can explore everything and change analysis parameters for their own
session. Only the editor (password) changes the stored inputs.

## Deploying (one time)

1. **Push this repository** to GitHub (`main` branch).
2. **Weekly data.** In the repository: *Actions* tab, enable workflows, open
   *weekly-data*, *Run workflow*. After about ten minutes a `data` branch
   exists. Optionally add the repository secret `SOCRATA_APP_TOKEN`
   (free, from data.cdc.gov) to raise the CDC rate limit.
3. **Google Sheet for expert inputs.**
   * Create an empty Google Sheet; copy its ID from the URL
     (`https://docs.google.com/spreadsheets/d/<ID>/edit`).
   * In Google Cloud Console create a project, enable the *Google Sheets
     API*, create a *service account*, and download its JSON key.
   * Share the sheet with the service account's email as *Editor*.
4. **Streamlit Community Cloud.** *Create app* from this repository, branch
   `main`, file `app.py`, Python 3.12 or 3.13. In *Advanced settings >
   Secrets* paste the entries of `.streamlit/secrets.toml.example` with your
   values (new Gemini key, editor password, sheet ID, service account).
5. Optional: give the weekly workflow the same sheet (repository secrets
   `SSDM_SHEET_ID` and `GCP_SERVICE_ACCOUNT` with the JSON key) so its
   update log uses your saved inputs.

The app sleeps after twelve hours without visitors; the first visitor wakes
it within a minute. GitHub disables scheduled workflows in a repository
without activity for sixty days; re enable it from the Actions tab if that
happens.

## Running locally

```bash
pip install -r requirements.txt
python scripts/refresh_data.py          # downloads the feeds into data/feeds
streamlit run app.py
```

## Reproducing a run

`python scripts/export_run.py --out run.xlsx` exports every input,
intermediate quantity and result of the current run with its manifest;
`--as-of YYYY-MM-DD` reconstructs the dashboard as it stood on a past date.
The Method Trace page shows the same chain step by step, and the SSDM
Integration page downloads the run manifest (settings, data week, inputs
fingerprint, code commit).

## Tests

`pip install -r requirements-dev.txt` then `python -m pytest tests` runs the verification suite (the weekly workflow runs it on every new week of data before publishing): an independent
implementation of every equation of the method compared with the engine on
random problems, the formal properties, the scenario and integration
quantities, the storage layer, and every page of the app.

## Structure

```text
app.py                      navigation, editor sign in, readiness gate
modules/
  mosdm_core.py             stratification core
  scenario_admission.py     weekly episode counting, marker magnitudes
  scenario_engine.py        registry, emphasis, tightening, scenario weights, stability
  ssdm_core.py              indicators, typology, integration
  agent_engine.py           the eight perspectives and the stratification call
  live_subcriteria.py       live criteria, history window, freshness
  live_connectors.py        readers over the CDC feeds
  feeds.py                  feed storage (local, legacy workbook, data branch)
  state.py                  inputs, settings, access, cache keys
  weight_manager.py         elicitation (fuzzy chain, respondents, aggregation)
  evidence_gate.py          external evidence for decision makers (information only)
  sensitivity.py            in app sensitivity analyses
  method_trace.py           the method step by step for the current run
  agent_view.py, scenario_view.py, ssdm_view.py, pages_e.py, controls.py, about.py, ui.py
  agents/                   profile builders of the eight perspectives
data/
  curated_seed.xlsx         curated evidence (master copy)
  scenario_emphasis.json    criteria emphasized by each scenario
scripts/
  refresh_data.py           CDC download
  export_run.py             full run export, weekly derived files, replay
tests/                      verification suite
.github/workflows/weekly-data.yml
```
