"""Record/subject-macro HR metrics, preserving original window-ID lag semantics."""
from collections import defaultdict
import math

import numpy as np

CORE_METRICS=('absolute','centered','lag1','lag2','lag4')


def score(rows,field):
    """Score already reference-qualified rows without reindexing original IDs.

    Subjects with no pairs at a lag are omitted from that lag's macro only.
    If no pairs exist anywhere, the lag error is None, never zero. Centering is
    an evaluation diagnostic, not a deployable calibration of predictions.
    """
    grouped=defaultdict(dict)
    for row in rows:
        person,record=str(row['subject']),str(row['record']);wid=int(row['window_id']);key=(person,record)
        if wid<0 or wid in grouped[key]:raise ValueError('Unique nonnegative original window IDs required')
        if not all(math.isfinite(float(row[k])) for k in ('label_bpm',field)):raise ValueError('Filter invalid references explicitly before scoring')
        grouped[key][wid]=row
    if not grouped:raise ValueError('Nonempty qualified reference population required')
    by_record={};per_person=defaultdict(list);counts={str(k):0 for k in (1,2,4)};time_gaps={str(k):[] for k in (1,2,4)}
    for (person,record),lookup in sorted(grouped.items()):
        ids=sorted(lookup);y=np.array([float(lookup[i]['label_bpm']) for i in ids]);p=np.array([float(lookup[i][field]) for i in ids]);error=p-y
        values=dict(absolute=float(abs(error).mean()),centered=float(abs(error-error.mean()).mean()),mse=float((error*error).mean()));record_counts={}
        for lag in (1,2,4):
            pairs=[(i,i+lag) for i in ids if i+lag in lookup];record_counts[str(lag)]=len(pairs);counts[str(lag)]+=len(pairs);values[f'lag{lag}']=float(np.mean([abs((float(lookup[j][field])-float(lookup[i][field]))-(float(lookup[j]['label_bpm'])-float(lookup[i]['label_bpm']))) for i,j in pairs])) if pairs else None
            for i,j in pairs:
                if 'start_seconds' in lookup[i] and 'start_seconds' in lookup[j]:
                    dt=float(lookup[j]['start_seconds'])-float(lookup[i]['start_seconds'])
                    if not math.isfinite(dt) or dt<=0:raise ValueError('Original window start times must increase')
                    time_gaps[str(lag)].append(dt)
        by_record[person+'/'+record]=dict(subject=person,record=record,windows=len(ids),pair_counts=record_counts,metrics=values);per_person[person].append(values)
    names=CORE_METRICS+('mse',);subjects={person:{name:float(np.mean([v[name] for v in values if v[name] is not None])) if any(v[name] is not None for v in values) else None for name in names} for person,values in per_person.items()};macro={name:float(np.mean([v[name] for v in subjects.values() if v[name] is not None])) if any(v[name] is not None for v in subjects.values()) else None for name in names};macro['rmse']=math.sqrt(macro['mse'])
    gaps={key:dict(count=len(v),minimum=min(v) if v else None,median=float(np.median(v)) if v else None,maximum=max(v) if v else None) for key,v in time_gaps.items()}
    return dict(subject_macro=macro,subject=subjects,records=by_record,pair_counts=counts,pair_time_gaps_seconds=gaps,coverage=dict(finite_windows=sum(len(v) for v in grouped.values()),records=len(grouped),subjects=len(subjects)),metric_subject_counts={name:sum(v[name] is not None for v in subjects.values()) for name in names})


def independent_macro(rows,field):
    """Second scalar reduction: enumerate unordered pairs and exact ID gaps."""
    people={str(r['subject']) for r in rows};per={};pairs={str(k):0 for k in (1,2,4)}
    def avg(values):return math.fsum(values)/len(values)
    for person in sorted(people):
        values=defaultdict(list)
        for record in sorted({str(r['record']) for r in rows if str(r['subject'])==person}):
            group=[r for r in rows if str(r['subject'])==person and str(r['record'])==record];errors=[float(r[field])-float(r['label_bpm']) for r in group];mean=avg(errors);values['absolute'].append(avg([abs(v) for v in errors]));values['centered'].append(avg([abs(v-mean) for v in errors]));values['mse'].append(avg([v*v for v in errors]));lag_errors=defaultdict(list)
            for i,left in enumerate(group):
                for right in group[i+1:]:
                    k=abs(int(right['window_id'])-int(left['window_id']))
                    if k in (1,2,4):lag_errors[k].append(abs((float(right[field])-float(left[field]))-(float(right['label_bpm'])-float(left['label_bpm']))))
            for k in (1,2,4):
                pairs[str(k)]+=len(lag_errors[k])
                if lag_errors[k]:values[f'lag{k}'].append(avg(lag_errors[k]))
        per[person]={k:avg(v) for k,v in values.items()}
    names=CORE_METRICS+('mse',);macro={k:avg([v[k] for v in per.values() if k in v]) if any(k in v for v in per.values()) else None for k in names};macro['rmse']=math.sqrt(macro['mse']);return macro,pairs
