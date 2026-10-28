# CodeForGeneratingDataset: In-Silico T1D Glucoregulatory Population Simulator

This repository contains the simulation pipeline and dataset generation codebase for **Type 1 Diabetes (T1D)** in-silico studies, conforming to **Roquemen-Echeverri et al., Sec. 2.2.1** (*A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes*) and matching the **HPC High-Performance Computing Dataset Architecture (SIM002, SIM004, SIM005)**.

The framework simulates the human glucoregulatory system using an 8-state single-hormone model (Hovorka model modified by Resalat et al. 2019, Oregon Health & Science University).

---

## 📌 Project Architecture

```text
CodeForGeneratingDataset/
├── all_subject_t1dexi_pilot_06222023.pkl    # Clinical pilot dataset from T1DEXI (44 virtual subjects)
├── datasetPop.py                            # Main dataset generator (HPC & Paper conforming)
├── build_dataset_splits.py                  # HPC File Count Verification & Split Manifest Generator
├── calc_metrics.py                          # 2-column clinical metrics evaluator (TIR, TAR, TBR, MG, HBGI, LBGI)
├── run_small_experiment.py                  # Quick runner for small cohort testing & immediate evaluation
├── correction_bolus.py                      # Smart bolus calculator & linear 4-h IOB (used in SIM005)
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore configuration
└── OHSUPOPsimulations/                      # Core simulation subpackage
    ├── __init__.py                          # Package initialization
    ├── create_SHP.py                        # Single-Hormone Plant: state-space matrices & steady-state solver
    ├── test_SHP.py                          # Meal absorption, rescue carbs, exercise, and sensor noise
    ├── SIM_generate_dosing_problems.py      # Dosing error generation (Basal & ICR misconfigurations, clean 24-h partition)
    ├── SIM_load_virtualpop_data.py          # Data extraction for meals and initial CGM conditions (N(mu, sigma))
    └── SIM_options.py                       # Simulation configuration & HPC dataset definitions
```

---

## 📊 Final Dataset Structure & HPC File Count Report

The complete dataset comprises **5,010 files** across **15 verified entries**, divided into three distinct simulation series:

```text
===================================================================================
                    FINAL DATASET FILE COUNT REPORT
                     HPC Dataset Verification Summary
===================================================================================
Dataset    Scenario / Folder              Data Split     File Count
-----------------------------------------------------------------------------------
SIM002     POPULATION                     TRAIN          270
SIM002     POPULATION                     TEST           75
SIM002     POPULATION                     VALIDATION     75
SIM002 Total                                             420
-----------------------------------------------------------------------------------
SIM004     boluslate_5min                 TRAIN          420
SIM004     boluslate_10min                TRAIN          420
SIM004     boluslate_15min                TRAIN          420
SIM004     boluslate_20min                TRAIN          420
SIM004     boluslate_25min                TRAIN          420
SIM004     boluslate_30min                TRAIN          420
SIM004     boluslate_35min                TRAIN          420
SIM004     boluslate_40min                TRAIN          420
SIM004     boluslate_45min                TRAIN          420
SIM004 Total                                             3,780
-----------------------------------------------------------------------------------
SIM005     bolusCorrectionFactor_28       TRAIN          270
SIM005     bolusCorrectionFactor_35       TRAIN          270
SIM005     bolusCorrectionFactor_42       TRAIN          270
SIM005 Total                                             810
-----------------------------------------------------------------------------------
GRAND TOTAL (15 verified entries)                        5,010
===================================================================================
```

### Dataset Distribution Summary
- **SIM002**: 420 files (8.38%)
- **SIM004**: 3,780 files (75.45%)
- **SIM005**: 810 files (16.17%)
- **Grand Total**: 5,010 files (100.0%)

---

## 🔬 Scientific & Mathematical Derivation

### 1. Cohort & Subject Partitioning
The virtual population uses **28 subjects** from the T1DEXI pilot study whose midnight CGM matches the published population mean of **$156 \pm 45\text{ mg/dL}$**:
- **TRAIN (18 subjects)**: `001003`, `001004`, `001008`, `001012`, `001016`, `001017`, `001026`, `001032`, `001033`, `001034`, `037001`, `037002`, `037003`, `037004`, `037006`, `037007`, `037008`, `037014`
- **TEST (5 subjects)**: `042006`, `042007`, `042011`, `042012`, `042013`
- **VALIDATION (5 subjects)**: `062005`, `064003`, `064005`, `064008`, `064011`

### 2. How File Counts Are Formed
For each subject in a scenario:
- **3 daily meal scenarios** are selected (each covering 7 days, filtered between 1–9 meals/day and 30–359 g/day total carbs).
- **5 initial glucose values** are drawn from $\mathcal{N}(\mu, \sigma)$ of the subject's midnight CGM.
- Therefore, each subject yields: $3 \times 5 = \mathbf{15\text{ simulation files}}$.

Across the splits:
- **TRAIN**: $18\text{ subjects} \times 15 = \mathbf{270\text{ files}}$
- **TEST**: $5\text{ subjects} \times 15 = \mathbf{75\text{ files}}$
- **VALIDATION**: $5\text{ subjects} \times 15 = \mathbf{75\text{ files}}$
- **Per-Scenario Total**: $270 + 75 + 75 = \mathbf{420\text{ files}}$

### 3. Purpose of Each Dataset Series
- **SIM002 (`POPULATION`)**: Baseline simulations with prandial insulin boluses administered at meal onset ($\text{delay} = 0\text{ min}$). Evaluated across TRAIN (270), TEST (75), and VALIDATION (75).
- **SIM004 (`boluslate_XXmin`)**: Augmentation simulations with boluses delayed from **5 to 45 minutes** in 5-minute increments (9 delays $\times$ 420 files = 3,780 files in TRAIN). These delayed traces capture rare postprandial extreme hyperglycemic events ($> 250\text{ mg/dL}$) as required in Section 2.2.1.
- **SIM005 (`bolusCorrectionFactor_XX`)**: Correction bolus sensitivity analysis across 3 correction factors: $28$ ($35 \times 0.8$), $35$, and $42$ ($35 \times 1.2$) on the 18 TRAIN subjects ($3 \times 270 = 810\text{ files}$).

### 4. Dosing Problems & Simulated Days
Each individual pickle file contains **11 complete 7-day trajectories** (1 normal + 10 clinical dosing problems across 4 daily timeframes: 7–11 AM, 11 AM–3 PM, 3–8 PM, 8 PM–7 AM):
- Rows per file: $11\text{ scenarios} \times 7\text{ days} \times 288\text{ samples/day} = \mathbf{22,176\text{ timesteps}}$
- SIM002 + SIM004 represent the paper's core 10 bolus timings:
  $$4,200\text{ files} \times 11\text{ dosing problems} \times 7\text{ days} = \mathbf{323,400\text{ simulated days}}$$

---

## 🚀 Execution Guide

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Generating the Datasets

#### Generate SIM002 (420 files: 270 TRAIN, 75 TEST, 75 VALIDATION)
```powershell
python datasetPop.py --dataset SIM002
```

#### Generate SIM004 (3,780 files: 9 delays in TRAIN)
```powershell
python datasetPop.py --dataset SIM004 --parallel
```

#### Generate SIM005 (810 files: 3 correction factors in TRAIN)
```powershell
python datasetPop.py --dataset SIM005 --parallel
```

#### Generate the Entire 5,010-File HPC Dataset
```powershell
python datasetPop.py --dataset ALL --parallel
```

### 3. Verify HPC File Counts & Generate Manifest
To inspect all generated files and produce the exact HPC verification summary table:
```powershell
python build_dataset_splits.py
```

### 4. Evaluate Clinical Metrics vs. Paper
To evaluate any generated dataset folder against the paper's published benchmarks:
```powershell
python calc_metrics.py simulation_results/SIM002/POPULATION/
```
Output:
```text
=================================================================
      SIMULATION RESULTS vs. PAPER (Roquemen-Echeverri et al.)
=================================================================
Metric                    | Your Simulation    | Paper (T1DSim_ODE)
-----------------------------------------------------------------
Time in Range (70-180)    | 75.7 ± 32.4 %      | 72.9 ± 35.2 %
Time Above Range (>180)   | 21.3 ± 33.4 %      | 23.1 ± 36.3 %
Time Below Range (<70)    | 3.0 ± 5.7 %        | 4.0 ± 8.2 %
Mean Glucose              | 133.2 ± 45.7 mg/dL | 132.6 ± 48.2 mg/dL
High BG Index (HBGI)      | 4.1 ± 5.8          | 4.2 ± 6.2
Low BG Index (LBGI)       | 1.4 ± 1.6          | 1.5 ± 1.7
=================================================================
```

### 5. Quick Test on Small Cohort
To run a fast simulation test on 3 representative patients (`001034` for median, `001033` for low, `037008` for high) without generating the full dataset:
```powershell
python run_small_experiment.py
```

---

## 📊 Data File Schema

Each `.pkl` file contains a Pandas DataFrame with the following columns:
- `time`: Simulation time index (5-minute intervals).
- `dosing_problem_id`: Identifier of dosing error scenario (0 = normal, 1–10 = dosing problems).
- `sim_day`, `sim_hour_of_day`: Day (1–7) and hour timestamping.
- `initCGM`: Starting blood glucose concentration ($\text{mg/dL}$).
- `input_insulin`: Total insulin administered ($\text{U/h}$).
- `input_meal_carbs`: Carbohydrate intake including rescue carbs ($\text{g}$).
- `output_glucose`: Ground-truth blood glucose concentration ($\text{mg/dL}$, $Q_1$ target for model training).
- `output_cgm`: Sensor CGM reading including noise and interstitial lag ($\text{mg/dL}$).
- `state_Q1` through `state_Ug`: Compartment state variables of the physiological model.
- `correction_bolus`: High-glucose corrective bolus doses administered ($\text{U/h}$).
- `subject_id`: Virtual subject identifier (6 digits).
- `meal_scenario`: Meal scenario identifier (0–2).
- `late_bolus_min`: Bolus timing delay in minutes (0–45).
- `meal_day_ids`: SHA-256 hash string of each 24-h daily meal vector (used for zero-leakage verification).
- `data_split`: Split identifier (`TRAIN`, `TEST`, or `VALIDATION`).

---

## 📚 References

- Roquemen-Echeverri V., Kushner T., Jacobs P.G., Mosquera-Lopez C. *A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes.* arXiv:2508.05705, 2025.
- Resalat N., El Youssef J., Tyler N., Castle J., Jacobs P.G. *A statistical virtual patient population for the glucoregulatory system in type 1 diabetes with integrated exercise model.* PLoS ONE 14(7): e0219454, 2019.
- Davidson P.C., Hebblewhite H.R., Steed R.D., Bode B.W. *Analysis of guidelines for basal-bolus insulin dosing: Basal insulin, correction factor, and carbohydrate-to-insulin ratio.* Endocrine Practice 14(9): 1095–1101, 2008.
