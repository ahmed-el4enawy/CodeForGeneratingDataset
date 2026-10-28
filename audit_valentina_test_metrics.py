#!/usr/bin/env python3
"""Read-only clinical audit of SIM002/POPULATION/TEST pickle files.

Calculates TIR/TAR/TBR, LBGI/HBGI, mean glucose on 5-hour, non-overlapping
windows (60 x 5-minute samples) and on complete 7-day segments. Reports
both output_glucose and output_cgm separately. Does not claim to reproduce
paper's exact aggregation protocol.

Usage: python audit_valentina_test_metrics.py --test-dir /path/SIM002/POPULATION/TEST --output-dir /path/audit_results

Only unpickle data supplied by a trusted source.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

STEPS=2016
WINDOW=60

def measures(g):
    x=np.asarray(g,dtype=float)
    if not np.isfinite(x).all() or np.any(x<=0):
        return None
    f=1.509*((np.log(x)**1.084)-5.381)
    risk=10.0*f*f
    return [100*float(np.mean(x>=70) - np.mean(x>180)),
            100*float(np.mean(x>180)),
            100*float(np.mean(x<70)),
            float(np.mean(np.where(f<0,risk,0))),
            float(np.mean(np.where(f>0,risk,0))),
            float(np.mean(x))]

NAMES=['TIR','TAR','TBR','LBGI','HBGI','Mean_Glucose']

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--test-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    root=a.test_dir.resolve(); out=a.output_dir.resolve()
    if not root.is_dir(): p.error('Test directory does not exist: '+str(root))
    if root==out or root in out.parents: p.error('Write results outside the original TEST folder')
    files=sorted(root.rglob('*.pkl'))
    if not files: p.error('No .pkl files found')
    out.mkdir(parents=True,exist_ok=True)
    stats=defaultdict(list)
    counter=defaultdict(int)
    perfile=[]
    for n,fp in enumerate(files,1):
        try:
            obj=pd.read_pickle(fp)
            df=obj['df'] if isinstance(obj,dict) and 'df' in obj else obj
            if not isinstance(df,pd.DataFrame): raise ValueError('Not a DataFrame')
            if len(df)%STEPS: raise ValueError('Row count not divisible by 2016')
            for col in ['output_glucose','output_cgm']:
                if col not in df.columns: raise ValueError('Missing '+col)
            for start in range(0,len(df),STEPS):
                part=df.iloc[start:start+STEPS]
                # Detect possible mixed dosing ID or interrupted trajectories.
                if 'dosing_problem_id' in part and part['dosing_problem_id'].nunique()!=1:
                    counter['mixed_dosing_segments']+=1
                counter['seven_day_segments']+=1
                for col in ['output_glucose','output_cgm']:
                    whole=measures(part[col].to_numpy())
                    if whole is not None: stats[(col,'7day')].append(whole)
                    else: counter['invalid_7day_'+col]+=1
                    values=part[col].to_numpy()
                    for pos in range(0,STEPS-WINDOW+1,WINDOW):
                        m=measures(values[pos:pos+WINDOW])
                        if m is not None: stats[(col,'5hour_nonoverlap')].append(m)
                        else: counter['invalid_5hour_'+col]+=1
            perfile.append({'file':str(fp.relative_to(root)),'rows':len(df),'segments':len(df)//STEPS,'error':''})
        except Exception as ex:
            counter['file_errors']+=1
            perfile.append({'file':str(fp.relative_to(root)),'rows':'','segments':'','error':str(ex)[:300]})
        if n%25==0 or n==len(files):print(f'Read {n}/{len(files)}',flush=True)
    with (out/'valentina_test_file_checks.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['file','rows','segments','error']);w.writeheader();w.writerows(perfile)
    rows=[]
    for (col,unit),values in sorted(stats.items()):
        arr=np.asarray(values)
        for i,name in enumerate(NAMES):
            rows.append({'signal':col,'aggregation':unit,'metric':name,'n':len(values),
                         'mean':round(float(np.mean(arr[:,i])),4),
                         'sd':round(float(np.std(arr[:,i],ddof=1)),4) if len(values)>1 else None})
    with (out/'valentina_test_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['signal','aggregation','metric','n','mean','sd']);w.writeheader();w.writerows(rows)
    report={'files_found':len(files),'counters':dict(counter),'notes':[
      'Metrics are averaged across individual windows, not across subjects.',
      'The 5-hour window aggregation uses non-overlapping 60-sample windows within each 2016-sample segment; trailing 36 samples are excluded.',
      'The paper may use a different window alignment, filtering and statistical aggregation. This is an audit, not an exact paper replication.',
      'TIR uses 70 <= glucose <= 180; TAR > 180; TBR < 70.',
      'LBGI/HBGI use standard Kovatchev transformation, valid only for glucose > 0.'
    ]}
    (out/'valentina_test_audit_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n',json.dumps(report,indent=2))
    print('\nClinical metrics (5-hour windows, output_glucose):')
    for rec in rows:
        if rec['signal']=='output_glucose' and rec['aggregation']=='5hour_nonoverlap':
            print(f"  {rec['metric']}: {rec['mean']} +/- {rec['sd']} (n={rec['n']})")
    print('\nReports:',out)
if __name__=='__main__':main()
