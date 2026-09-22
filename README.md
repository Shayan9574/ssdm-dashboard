# SSDM Public Health Decision Support Dashboard

A public health prioritization dashboard built with Python and Streamlit. The system combines 365 day macro baselines with live weekly CDC surveillance feeds across four target pathogens (COVID-19, Influenza, RSV, Meningococcal disease), ranking them with the updated MOSDM stratification and probability weighted SSDM described in the accompanying methods document.

## Architecture

* **This repository** holds all code plus `data/curated_seed.xlsx`, the eight curated sheets (Baseline Data, Historical Outbreaks & Surges, Data Dictionary, Citations, RESP-NET, Weekly Hospital Respiratory Adm, and the two provisional COVID-19 death sheets). Total under 2 megabytes.
* **Google Drive** holds the full master workbook `Research Data.xlsx` (about 42 megabytes). It is never committed: the six surveillance sheets (NHSN, NSSP, NREVSS, NWSS, CFA, NNDSS) are rebuilt from the public CDC Socrata APIs by `scripts/refresh_data.py`.
* **Google Colab** runs the app: the launcher notebook clones this repository, mounts Drive, refreshes the workbook when it is older than seven days, and serves Streamlit through a tunnel.

## Quickstart (Colab)

Open `notebooks/launcher.ipynb` in Google Colab and run all cells. The notebook handles cloning, Drive mounting, the staleness check, and the tunnel. The Gemini key is read from Colab Secrets (`GEMINI_API_KEY`); never commit keys.

## Quickstart (local)

```bash
git clone https://github.com/<your-username>/ssdm-dashboard.git
cd ssdm-dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/refresh_data.py   # builds data/Research Data.xlsx from the seed plus CDC APIs
streamlit run app.py
```

## Project structure

```text
ssdm-dashboard/
├── app.py                       # Main Streamlit application
├── diagnose.py                  # Diagnostic and self test suite
├── data/
│   └── curated_seed.xlsx        # Curated sheets (committed; the master workbook is not)
├── modules/
│   ├── load_data.py             # Data loading and citation indexing
│   ├── utils_numeric.py         # Parsing utilities
│   ├── agent_m_data.py          # Decision matrix and live data merger
│   ├── live_connectors.py       # Real time CDC time series connectors
│   ├── scenarios.py             # Scenario definitions
│   ├── status_banner.py         # Pipeline status banner
│   ├── genai.py / evidence.py   # AI and web search services
│   └── agents/                  # Eight domain analytical agents
├── scripts/
│   └── refresh_data.py          # CDC Socrata ETL
└── notebooks/
    └── launcher.ipynb           # Colab launcher (added in the next step)
```
