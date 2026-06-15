# DRMC OPD Patient Flow Simulation

A comprehensive Discrete Event Simulation (DES) model for the Davao Regional Medical Center (DRMC) Outpatient Department patient flow analysis.

## Overview

This simulation models the complete patient journey through DRMC's OPD, from early morning arrival (5:00 AM) through gate screening, verification/vitals, registration, sub-clinic routing, doctor consultation, and post-consultation services.

## Features

- **Patient Arrival Modeling**: Poisson arrival process with four distinct phases
- **Priority Patient Support**: Non-preemptive priority queuing for vulnerable populations
- **Multi-Stage Process**: Models all stages from pre-opening queue to final exit
- **Doctor Freeze Events**: Simulates physician interruptions for ER/ward duties
- **Interactive Dashboard**: Streamlit-based web UI with real-time visualization

## Installation

```bash
pip install -r requirements.txt
streamlit run DRMC_OPD_Simulation.py
```

Then open http://localhost:8501 in your browser.

## Key Parameters

- **Daily Volume**: 500-1000 patients
- **Arrival Phases**: Dawn Surge, Peak Congestion, Midday, Afternoon
- **Processing Times**: Triangular distributions for all stages
- **Intervention Scenarios**: A (Front-End), B (Consultation Caps), C (No-Show), D (Mid-Week)

## KPIs Tracked

- Pre-Opening Queue Build-up
- Waiting Times by Stage (Gate, Registration, Clinic, Consultation)
- Total OPD Length of Stay
- Doctor Freeze Events
- Resource Utilization Rates
- Patient Throughput

