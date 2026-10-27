# CodeForGeneratingDataset: In-Silico T1D Glucoregulatory Population Simulator

This repository contains the simulation pipeline and dataset generation codebase for **Type 1 Diabetes (T1D)** in-silico studies. It models the glucoregulatory system of virtual patients using an 8-state single-hormone model (Hovorka glucoregulatory model modified by Resalat et al., Oregon Health & Science University).

The pipeline generates high-resolution time-series datasets of continuous glucose monitoring (CGM), insulin delivery, carbohydrate intake, and physiological compartment states under varying scenarios, including **10 clinically realistic insulin dosing problem profiles**.

---

## 📌 Project Architecture

```text
CodeForGeneratingDataset/
├── all_subject_t1dexi_pilot_06222023.pkl    # Clinical pilot dataset from T1DEXI (44 virtual subjects)
├── correction_bolus.py                      # Smart bolus calculator & Insulin On Board (IOB)
├── OHSU_population_simulation.py            # Main dataset generation entry point & simulation loop
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore configuration
└── OHSUPOPsimulations/                      # Core simulation subpackage
    ├── __init__.py                          # Package initialization
    ├── create_SHP.py                        # Single-Hormone Plant: state-space matrices & steady-state solver
    ├── test_SHP.py                          # Meal absorption, rescue carbs, exercise, and sensor noise
    ├── SIM_generate_dosing_problems.py      # Dosing error generation (Basal & ICR misconfigurations)
    ├── SIM_load_virtualpop_data.py          # Data extraction for meals and initial CGM conditions
    └── SIM_options.py                       # Simulation configuration and hyperparameter settings
```

---

## 🔬 Scientific & Physiological Background

- **Glucoregulatory Model**: 8-state ODE model representing:
  - Glucose kinetics: $Q_1, Q_2$
  - Insulin absorption & kinetics: $S_1, S_2, I$
  - Insulin dynamics / remote compartments: $x_1, x_2, x_3$
- **Meal Absorption**: Hovorka two-compartment gut absorption model.
- **Rescue Carbohydrates**: Automatic rescue carb administration (20g) upon hypoglycemia detection ($< 70\text{ mg/dL}$).
- **Correction Boluses**: Linear-decay Insulin On Board (IOB) calculator providing corrective insulin when glucose exceeds hyperglycemia thresholds ($200\text{ mg/dL}$ daytime, $280\text{ mg/dL}$ nighttime).
- **Sensor Noise**: Interstitial glucose delay + AR(2) autoregressive sensor error model (Vettoretti et al., 2019).
- **Exercise Response**: Hernandez aerobic exercise model solved via symbolic ODEs (`sympy`).

---

## ⚙️ Dosing Problems Simulated

The simulator introduces 10 distinct clinical dosing problems across 4 daily timeframes (7–11 AM, 11 AM–3 PM, 3–8 PM, 8 PM–7 AM):
1. **Post-prandial hypoglycemia**: ICR low (over-bolusing).
2. **Post-prandial severe hypoglycemia**: ICR very low (severe over-bolusing).
3. **Post-prandial hyperglycemia**: ICR high (under-bolusing).
4. **Post-prandial severe hyperglycemia**: ICR very high (severe under-bolusing).
5. **Aggressive basal**: Basal rate elevated.
6. **Insufficient basal**: Basal rate deficient.
7. **Basal high, ICR high**.
8. **Basal high, ICR low**.
9. **Basal low, ICR high**.
10. **Basal low, ICR low**.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12+

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Simulation

Run the main dataset generator script:

```bash
python OHSU_population_simulation.py
```

### Configuration
Simulation parameters can be tuned in `OHSUPOPsimulations/SIM_options.py`:
- `SIMULATION_SERIAL`: Run sequentially (`True`) or parallelized across CPU cores with `multiprocessing` (`False`).
- `SIMULATION_N_DAYS_WEEKS`: Number of days per simulated scenario.
- `SIMULATION_TS_MINUTES`: Sampling period (default: `5` minutes).
- `SIMULATION_DOSING_PROBLEMS_SELECTED`: Active dosing problems to evaluate.
- `SIMULATION_CORRECTION_BOLUS`: Enable/disable smart bolus correction.

---

## 📊 Output Data Format

The output pickle files contain pandas DataFrames structured with the following columns:
- `time`: Simulation time index (5-minute intervals).
- `dosing_problem_id`: Identifier of dosing error scenario (0 = normal, 1–10 = dosing problems).
- `sim_day`, `sim_hour_of_day`: Day and hour timestamping.
- `input_insulin`: Total insulin administered ($U/h$).
- `input_meal_carbs`: Carbohydrate intake from meals and rescue carbs ($g$).
- `output_glucose`: Ground-truth blood glucose concentration ($mg/dL$).
- `output_cgm`: Sensor CGM reading including noise and interstitial lag ($mg/dL$).
- `state_Q1` through `state_Ug`: Physiological states of the compartment model.
- `correction_bolus`: High-glucose corrective bolus doses administered.

---

## 📚 References & Acknowledgments

- **Model Formulation**: Resalat N, El Youssef J, Tyler N, Castle J, Jacobs PG. *A statistical virtual patient population for the glucoregulatory system in type 1 diabetes with integrated exercise model.* PLoS ONE, 2019.
- **Sensor Noise**: Vettoretti M, Facchinetti A, Sparacino G, Cobelli C. *Type 1 Diabetes Patient by In Silico Simulation: A Review of Glucose Models for Assessment of Sensor-Augmented Pump and Artificial Pancreas Technologies.* Sensors, 2019.
- **Original Research**: Biomedical Engineering Department, Oregon Health & Science University (OHSU).