"""
12 Public Health Decision Scenarios for SSDM Multi-Criteria Weighting.
"""

SCENARIOS_12 = [
    {
        "id": "scen_1",
        "name": "1. Baseline / Equal Weights",
        "description": "Standard balanced surveillance where all subcriteria are weighted equally.",
        "emphasis": []
    },
    {
        "id": "scen_2",
        "name": "2. Hospital Surge & Inpatient Strain",
        "description": "Severe facility overload prioritizing hospital admission rates, ICU volume, and inpatient stay duration.",
        "emphasis": ["hospitalization", "icu", "stay", "occupancy", "admissions"]
    },
    {
        "id": "scen_3",
        "name": "3. Acute Outbreak Velocity & Rapid Spread",
        "description": "Early exponential growth prioritizing reproduction numbers (Rt, R0) and outbreak growth probability.",
        "emphasis": ["rt", "r0", "growth", "positivity", "velocity"]
    },
    {
        "id": "scen_4",
        "name": "4. High Contagion & Community Transmission",
        "description": "Focus on highly infectious pathogens with high household attack rates and asymptomatic spread.",
        "emphasis": ["transmission", "attack", "asymptomatic", "incidence", "positivity"]
    },
    {
        "id": "scen_5",
        "name": "5. Vulnerable Populations & Inequity Focus",
        "description": "Focus on protecting pediatric populations, seniors (65+), and high-disparity demographic groups.",
        "emphasis": ["pediatric", "geriatric", "disparity", "equity", "vulnerable"]
    },
    {
        "id": "scen_6",
        "name": "6. Economic & Healthcare Cost Burden",
        "description": "Resource allocation prioritizing reduction of direct medical expenditures and societal impact.",
        "emphasis": ["cost", "economic", "direct", "svi"]
    },
    {
        "id": "scen_7",
        "name": "7. Pediatric Critical Care Surge",
        "description": "Targeted surge in pediatric hospitalizations and intensive care units.",
        "emphasis": ["pediatric", "icu", "hospitalization"]
    },
    {
        "id": "scen_8",
        "name": "8. Geriatric Severe Illness & Lethality",
        "description": "Surge among older adults with high case fatality and severe complication rates.",
        "emphasis": ["geriatric", "fatality", "complication", "cfr"]
    },
    {
        "id": "scen_9",
        "name": "9. Vaccine Shortage / Immune Evasion",
        "description": "Evasion of vaccine protection requiring emphasis on non-pharmaceutical interventions and therapeutics.",
        "emphasis": ["vaccine", "effectiveness", "coverage", "npi"]
    },
    {
        "id": "scen_10",
        "name": "10. Antimicrobial Resistance (AMR) Crisis",
        "description": "Pathogens exhibiting high drug resistance and narrow therapeutic intervention windows.",
        "emphasis": ["resistance", "amr", "treatment", "window", "onset"]
    },
    {
        "id": "scen_11",
        "name": "11. Rapid Clinical Deterioration",
        "description": "Urgent life-threatening pathogens with very few days from onset to severe illness.",
        "emphasis": ["onset", "window", "complication", "fatality"]
    },
    {
        "id": "scen_12",
        "name": "12. Dual Winter Respiratory Wave",
        "description": "Concurrent winter waves of COVID-19, Influenza, and RSV stressing both emergency departments and ICUs.",
        "emphasis": ["hospitalization", "icu", "ed", "positivity", "incidence"]
    }
]

def get_scenario_by_name(name: str):
    return next((s for s in SCENARIOS_12 if s["name"] == name), SCENARIOS_12[0])