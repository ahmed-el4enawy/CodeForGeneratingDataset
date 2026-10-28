# CodeForGeneratingDataset: In-Silico T1D Glucoregulatory Population Simulator

This repository contains the simulation pipeline and dataset generation codebase for **Type 1 Diabetes (T1D)** in-silico studies, conforming to **Roquemen-Echeverri et al., Sec. 2.2.1** (*A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes*).

It models the glucoregulatory system of virtual patients using an 8-state single-hormone model (Hovorka glucoregulatory model modified by Resalat et al. 2019, Oregon Health & Science University).

---

## 📌 Project Architecture

```text
CodeForGeneratingDataset/
├── all_subject_t1dexi_pilot_06222023.pkl    # Clinical pilot dataset from T1DEXI (44 virtual subjects)
├── datasetPop.py                            # Main dataset generator & simulation loop (paper-conforming)
├── build_dataset_splits.py                  # 60/20/20 train/val/test split without meal-day leakage + rare-event flags
├── correction_bolus.py                      # Smart bolus calculator & linear IOB (optional, off by default)
├── OHSU_population_simulation.py            # Entry point forwarding to datasetPop.py
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore configuration
└── OHSUPOPsimulations/                      # Core simulation subpackage
    ├── __init__.py                          # Package initialization
    ├── create_SHP.py                        # Single-Hormone Plant: state-space matrices & steady-state solver
    ├── test_SHP.py                          # Meal absorption, rescue carbs, exercise, and sensor noise
    ├── SIM_generate_dosing_problems.py      # Dosing error generation (Basal & ICR misconfigurations, clean 24-h partition)
    ├── SIM_load_virtualpop_data.py          # Data extraction for meals and initial CGM conditions (N(mu, sigma))
    └── SIM_options.py                       # Simulation configuration and hyperparameter settings
```

---

## 🔬 Paper Specifications & Implementation (Sec. 2.2.1)

| Feature | Paper Specification | Implementation |
|---|---|---|
| **Simulation Duration** | 7-day simulations started at midnight | `SIMULATION_N_DAYS_WEEKS = 7`; 2016 timesteps (5-min TS) |
| **Meal Scenarios** | 7 daily meal scenarios per 7-day simulation, 1-9 meals/day, 30-359 g/day total carbs | `SIM_load_virtualpop_data.py` (`SIMULATION_count_min/max`, `SIMULATION_amount_min/max`) |
| **Bolus Timing Delays** | Meal onset and delayed 5 to 45 min (5-min incremental steps) | `SIMULATION_LATE_BOLUS_MIN = range(0, 50, 5)` (10 delays); outputs in `boluslate_XXmin/` |
| **Under- / Over-Dosing** | 10 dosing problems + 1 normal (dosing problem id 0) | 11 dosing scenarios per `(meal_scenario, initCGM)` run |
| **Basal Delivery** | Delivered every 5 minutes | `Ts = 5` min constant basal infusion rate |
| **Initial Glucose** | 5 initial glucose values drawn from $\mathcal{N}(\mu, \sigma)$ of midnight CGM ($156 \pm 45\text{ mg/dL}$) | `get_initCGM` draws from midnight distribution |
| **Dataset Scale** | **323,400 days** total | $28\text{ subjects} \times 3\text{ meal scen} \times 5\text{ initCGM} \times 11\text{ dosing} \times 10\text{ timings} \times 7\text{ days} = 323,400\text{ days}$ |
| **Data Partitioning** | 60% train, 20% validation, 20% test without daily meal scenario leakage | `build_dataset_splits.py` connected-component meal clustering |
| **Rare Events** | Flag traces with TBR > 20% or glucose > 250 mg/dL > 40% | Flagged per simulation in `dataset_manifest.json` |

---

## ⚙️ Dosing Problems Simulated

The simulator introduces 10 distinct clinical dosing problems across 4 cleanly partitioned daily timeframes (7–11 AM, 11 AM–3 PM, 3–8 PM, 8 PM–7 AM):
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

## 🚀 Setup & Execution

### 1. Prerequisites & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
Run the main simulation generator:
```bash
python datasetPop.py
```
*(Or in parallel mode by setting `SIMULATION_SERIAL = False` in `OHSUPOPsimulations/SIM_options.py`)*

### 3. Build Leakage-Free Dataset Splits & Manifest
Generate the 60/20/20 train/validation/test splits and rare-event manifest:
```bash
python build_dataset_splits.py
# or provide custom path:
python build_dataset_splits.py simulation_results/SIM005/POPULATION/
```

---

## 📊 Output Data Format

Each pickled file corresponds to one `(meal_scenario, initCGM)` pair containing all 11 dosing scenarios (22,176 rows):
- `time`: Simulation time index (5-minute intervals).
- `dosing_problem_id`: Identifier of dosing error scenario (0 = normal, 1–10 = dosing problems).
- `sim_day`, `sim_hour_of_day`: Day (1–7) and hour timestamping.
- `initCGM`: Initial glucose value drawn from midnight CGM.
- `input_insulin`: Total insulin administered ($U/h$).
- `input_meal_carbs`: Carbohydrate intake from meals and rescue carbs ($g$).
- `output_glucose`: Ground-truth blood glucose concentration ($mg/dL$, $Q_1$ target for model training).
- `output_cgm`: Sensor CGM reading including noise and interstitial lag ($mg/dL$).
- `state_Q1` through `state_Ug`: Physiological states of the compartment model.
- `correction_bolus`: High-glucose corrective bolus doses administered.
- `subject_id`: Virtual subject identifier.
- `meal_scenario`: Meal scenario identifier (0–2).
- `late_bolus_min`: Bolus timing delay in minutes (0–45).
- `meal_day_ids`: Hash string of each 24-h daily meal vector (used for zero-leakage data splitting).

---

## 📚 References

- Roquemen-Echeverri V., Kushner T., Jacobs P.G., Mosquera-Lopez C. *A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes.* arXiv:2508.05705, 2025.
- Resalat N., El Youssef J., Tyler N., Castle J., Jacobs P.G. *A statistical virtual patient population for the glucoregulatory system in type 1 diabetes with integrated exercise model.* PLoS ONE 14(7): e0219454, 2019.
- Davidson P.C., Hebblewhite H.R., Steed R.D., Bode B.W. *Analysis of guidelines for basal-bolus insulin dosing: Basal insulin, correction factor, and carbohydrate-to-insulin ratio.* Endocrine Practice 14(9): 1095–1101, 2008.
