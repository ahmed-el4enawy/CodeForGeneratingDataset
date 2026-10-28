#!/usr/bin/env python3
"""Read-only audit of the authors' simulated T1D dataset.

Example:
  python audit_population_dataset.py \
    --root /nfs/slurm/cugp012/PopulationLevel_PIML_Dataset/GeneratedDataset/OHSUSimPopulation \
    --output-dir /nfs/slurm/cugp012/PopulationLevel_PIML_Dataset/audit_results

WARNING: Only unpickle files from a source you trust; Python pickle can execute code.
This script reads each .pkl file and writes aggregate audit artifacts only.
It NEVER changes the original files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sys

try:
    import pandas as pd
except ImportError:
    raise SystemExit("Requires pandas: python -m pip install 'pandas<2'")

TRAIN_IDS = set('001003 001004 001008 001012 001016 001017 001026 001032 001033 001034 037001 037002 037003 037004 037006 037007 037008 037014'.split())
TEST_IDS = set('042006 042007 042011 042012 042013'.split())
VALID_IDS = set('062005 064003 064005 064008 064011'.split())
SUBJECT_PATTERN = re.compile(r'(?:S_)?(\d{6})mealScenario_')
EXPECTED_SAMPLES = 2016
READER_STATES_FROM_HANDOFF = {'output_cgm', 'state_Q2','state_S1','state_S2','state_I','state_X1','state_X2','state_X3','state_Ug','state_Ug2'}
READER_INPUTS_FROM_HANDOFF = {'input_insulin','input_meal_carbs'}


def subject_from_filename(name):
    match = SUBJECT_PATTERN.search(name)
    return match.group(1) if match else None


def get_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get('df'), pd.DataFrame):
        return obj['df']
    raise ValueError('expected pandas DataFrame or dict containing DataFrame under df')


def inspect(path):
    df = get_df(pd.read_pickle(path))
    total_rows = len(df)
    if 'dosing_problem_id' in df.columns:
        # The original generator produces one 2016-row trajectory per DP identifier.
        groups = df.groupby('dosing_problem_id', sort=False, dropna=False)
        group_desc = []
        for pid, group in groups:
            pos = None
            if 'correction_bolus' in group.columns:
                values = pd.to_numeric(group['correction_bolus'], errors='coerce')
                pos = bool(values.fillna(0).sum() > 0)
            group_desc.append((str(pid),len(group),pos))
    elif 'event' in df.columns:
        group_desc = []
        for eid, group in df.groupby('event', sort=False, dropna=False):
            pos = None
            if 'correction_bolus' in group.columns:
                pos = bool(pd.to_numeric(group['correction_bolus'],errors='coerce').fillna(0).sum()>0)
            group_desc.append((str(eid),len(group),pos))
    else:
        group_desc = []
        # Do not invent event boundaries when neither identifier exists.
    return {
        'rows': total_rows,
        'events': len(group_desc),
        'good_events': sum(n == EXPECTED_SAMPLES for _,n,_ in group_desc),
        'bad_events': sum(n != EXPECTED_SAMPLES for _,n,_ in group_desc),
        'correction_positive': sum(v is True for _,_,v in group_desc),
        'correction_zero': sum(v is False for _,_,v in group_desc),
        'correction_unavailable': sum(v is None for _,_,v in group_desc),
        'columns': set(df.columns),
        'dp_ids': {pid for pid,_,_ in group_desc},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',required=True,type=Path,help='OHSUSimPopulation directory')
    parser.add_argument('--output-dir',required=True,type=Path,help='New output directory for aggregate reports')
    args=parser.parse_args()
    root=args.root.expanduser().resolve()
    output=args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error('Dataset root does not exist: '+str(root))
    if output == root or root in output.parents:
        parser.error('Place audit output OUTSIDE OHSUSimPopulation to preserve the source tree')
    files=sorted(p for p in root.rglob('*.pkl') if p.is_file())
    if not files:
        parser.error('No .pkl files found beneath '+str(root))
    output.mkdir(parents=True,exist_ok=True)
    summaries=defaultdict(lambda:{'files':0,'rows':0,'events':0,'good_events':0,'bad_events':0,
        'correction_positive':0,'correction_zero':0,'correction_unavailable':0,
        'subjects':set(),'columns':None,'dp_ids':set(), 'schema_variations':0,'read_errors':0})
    err=[]
    file_csv=output/'per_file_summary.csv'
    with file_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','subject_id','rows','events','good_events','bad_events',
            'correction_positive','correction_zero','correction_unavailable','missing_reader_columns','error'])
        writer.writeheader()
        for idx,p in enumerate(files,1):
            rel=p.relative_to(root)
            folder=rel.parent.as_posix()
            acc=summaries[folder]
            acc['files']+=1
            sub=subject_from_filename(p.name)
            if sub: acc['subjects'].add(sub)
            rec={'path':rel.as_posix(),'subject_id':sub or '', 'error':''}
            try:
                info=inspect(p)
                for key in ['rows','events','good_events','bad_events','correction_positive','correction_zero','correction_unavailable']:
                    acc[key]+=info[key]
                    rec[key]=info[key]
                acc['dp_ids'].update(info['dp_ids'])
                if acc['columns'] is None: acc['columns']=info['columns']
                elif acc['columns']!=info['columns']: acc['schema_variations']+=1
                rec['missing_reader_columns']=';'.join(sorted((READER_STATES_FROM_HANDOFF|READER_INPUTS_FROM_HANDOFF)-info['columns']))
            except Exception as exc:
                acc['read_errors']+=1
                rec['error']=str(exc)[:300]
                err.append({'path':str(rel),'error':str(exc)[:500]})
            writer.writerow(rec)
            if idx%250==0 or idx==len(files):
                print(f'Checked {idx:,}/{len(files):,} files',flush=True)

    with (output/'per_folder_summary.csv').open('w',newline='',encoding='utf-8') as f:
        headers=['folder','files','rows','events','good_events','bad_events','correction_positive',
                 'correction_zero','correction_unavailable','unique_subjects','train_id_count','validation_id_count',
                 'test_id_count','other_id_count','schema_variations','read_errors', 'missing_columns']
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader()
        for folder, acc in sorted(summaries.items()):
            cols=acc['columns'] or set()
            ids=acc['subjects']
            writer.writerow({
                'folder':folder,'files':acc['files'],'rows':acc['rows'],'events':acc['events'],
                'good_events':acc['good_events'],'bad_events':acc['bad_events'],
                'correction_positive':acc['correction_positive'],'correction_zero':acc['correction_zero'],
                'correction_unavailable':acc['correction_unavailable'],
                'unique_subjects':len(ids),'train_id_count':len(ids & TRAIN_IDS),
                'validation_id_count':len(ids & VALID_IDS),'test_id_count':len(ids & TEST_IDS),
                'other_id_count':len(ids - TRAIN_IDS - VALID_IDS - TEST_IDS),
                'schema_variations':acc['schema_variations'],'read_errors':acc['read_errors'],
                'missing_columns':';'.join(sorted((READER_STATES_FROM_HANDOFF|READER_INPUTS_FROM_HANDOFF)-cols))})

    family=defaultdict(lambda:{'files':0,'events':0,'retained':0,'folders':0})
    selection={'sim002_train':0,'sim002_validation':0,'sim002_test':0,
               'sim004_train':0,'sim005_train_candidates':0,'sim005_train_retained':0}
    leakage=[]
    for folder,acc in sorted(summaries.items()):
        parts=folder.upper().split('/')
        sim=next((x for x in parts if x in ('SIM002','SIM004','SIM005')),None)
        if sim:
            fam=family[sim];fam['files']+=acc['files'];fam['events']+=acc['events'];fam['folders']+=1
            if sim=='SIM005': fam['retained']+=acc['correction_positive']
        if 'TRAIN' in parts and (acc['subjects']& (VALID_IDS|TEST_IDS)):
            leakage.append({'folder':folder,'validation_subjects':len(acc['subjects']&VALID_IDS),
                            'test_subjects':len(acc['subjects']&TEST_IDS)})
        for split,label in [('TRAIN','train'),('VALIDATION','validation'),('TEST','test')]:
            if split in parts and sim=='SIM002':selection[f'sim002_{label}']+=acc['events']
        if sim=='SIM004' and 'TRAIN' in parts:selection['sim004_train']+=acc['events']
        if sim=='SIM005' and 'TRAIN' in parts:
            selection['sim005_train_candidates']+=acc['events']
            selection['sim005_train_retained']+=acc['correction_positive']

    selection['train_after_sim005_filter']=selection['sim002_train']+selection['sim004_train']+selection['sim005_train_retained']
    report={
        'dataset_root':str(root),'files_checked':len(files),'expected_samples_per_event':EXPECTED_SAMPLES,
        'family_totals':dict(family), 'training_reader_composition':selection,
        'sim005_retention_definition':'event sum of correction_bolus > 0',
        'possible_split_leakage_folders':leakage,
        'paper_reported_total_seven_day_trajectories':46200,
        'reference_total_gap_from_training_reader_count':46200-selection['train_after_sim005_filter'],
        'errors':err[:200], 'total_errors':len(err),
        'cautions':[
            'Paper total development traces should not automatically be equated with TRAIN-only reader output.',
            'State Ug2 requirement is from the handoff brief; check it directly against the original read_dataset.py.',
            'Group by dosing_problem_id assumes each saved file has exactly one trajectory per dosing ID.',
            'This script does not test duplicated meal-day histories or exact numeric reproducibility.',
            'Files use pickle: do not run on unknown or untrusted files.'
        ]
    }
    with (output/'audit_summary.json').open('w',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print('\nFINAL SUMMARY')
    print(json.dumps({k:report[k] for k in ('files_checked','family_totals','training_reader_composition','possible_split_leakage_folders','total_errors')},indent=2))
    print('\nReports:', output)

if __name__=='__main__':
    main()
