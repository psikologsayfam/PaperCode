"""Reconstruct a two-hour ICU admission cohort from HiRID raw v1.1.1.

Both measurement and electronic entry timestamps are checked against
the landmark. The source feature table and event definitions are validated.
"""
import io, json, os, re, shutil, tarfile, sys, traceback
from collections import Counter
from pathlib import Path
import pandas as pd, numpy as np
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
ROOT = Path(os.environ.get('NIV_RESEARCH_ROOT', 'D:\\ML\\research_campaign'))
BASE = ROOT / 'datasets' / 'HiRID'
ORIGINAL = BASE / 'drive_shared_sources'
RAW = BASE / 'raw_official_v1_1_1'
OUT = ROOT / 'runs' / 'NIV_R6_CLEAN_LANDMARK_BUILD' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1'))
REPORT = {'status': 'INITIALIZED', 'dataset': 'HiRID v1.1.1 official raw', 'landmark': 'ICU admission + exactly 2h', 'feature_gate': '0 <= datetime and entertime <= 2h', 'events': 'recorded NIV to invasive ventilation via variable 15001552', 'source': 'model final draft reference is NOT exact original _final.csv'}

def save():
    (OUT / 'SUMMARY.json').write_text(json.dumps(REPORT, indent=2, default=str), encoding='utf-8')
    print('CLEAN_BUILD ' + json.dumps(REPORT, default=str), flush=True)

def key_time(df, key, adm):
    return (df[key] - df[adm]).dt.total_seconds() / 3600

def main():
    with tarfile.open(RAW / 'reference_data.tar.gz', 'r:gz') as tar:
        gen = pd.read_csv(io.BytesIO(tar.extractfile('general_table.csv').read()), low_memory=False)
        varref = pd.read_csv(io.BytesIO(tar.extractfile('hirid_variable_reference.csv').read()), low_memory=False)
    gen['patientid'] = pd.to_numeric(gen.patientid, errors='raise').astype('int64')
    gen['admissiontime'] = pd.to_datetime(gen.admissiontime, errors='raise')
    ref = gen.set_index('patientid')
    src = pd.read_csv(ORIGINAL / '0_2 merged_respiratory_var_names_data 01 03 2024.csv', low_memory=False)
    old = pd.read_csv(ORIGINAL / '0_2 merged_respiratory_modelling final draft.csv', low_memory=False)
    modelids = set(pd.to_numeric(old.patientid).astype('int64'))
    eligible = src.loc[src.patientid.isin(modelids) & src['ventilation type'].eq('Non-Invasive')].copy()
    ids = set(eligible.patientid.astype('int64'))
    if len(ids) != 1424:
        raise ValueError('Expected 1424 source NIV patients, not ' + str(len(ids)))
    if not ids.issubset(set(ref.index)):
        raise ValueError('Official reference patient IDs missing')
    REPORT['source_eligible_n'] = len(ids)
    REPORT['official_reference_n'] = len(ref)
    variable_name = {int(vid): name for vid, name in json.loads(Path(__file__).with_name('hirid_variable_map.json').read_text(encoding='utf-8')).items()}
    model_columns = [c for c in old if c not in ('patientid', 'Outcome', 'sex', 'age')]
    feature_columns = [c for c in model_columns if c.endswith(('_first', '_last', '_mean')) and (not any((c.startswith(z) for z in ('Ventilator mode_', 'PO2 FiO2 ratio_', 'Cumulative Fluid balance_', 'Lung complience_')))) and (c.rsplit('_', 1)[0] in set(variable_name.values()))]
    omitted = [c for c in model_columns if c not in feature_columns]
    if len(feature_columns) < 25:
        raise RuntimeError('Too few direct raw predictor columns; original naming mismatch')
    approved_ids = {vid for vid, name in variable_name.items() if any((c.rsplit('_', 1)[0] == name for c in feature_columns))}
    REPORT.update({'direct_raw_feature_columns': len(feature_columns), 'planned_predictors': len(feature_columns) + 2, 'excluded_unvalidated_derived_or_mode_columns': omitted, 'mapped_raw_variable_ids': len(approved_ids)})
    ref_admission = ref.admissiontime
    allowed = pa.array(sorted(ids), type=pa.int32())
    wanted = pa.array(sorted(approved_ids | {15001552}), type=pa.int32())
    kept = []
    airway = []
    sel_count = 0
    total = 0
    parts = 0
    with tarfile.open(RAW / 'observation_tables_parquet.tar.gz', 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.parquet'):
                continue
            parts += 1
            f = archive.extractfile(member)
            if f is None:
                raise RuntimeError('Missing observation member ' + member.name)
            pqfile = pq.ParquetFile(pa.BufferReader(f.read()))
            cols = ['patientid', 'variableid', 'value', 'datetime', 'entertime', 'status']
            for batch in pqfile.iter_batches(batch_size=175000, columns=cols):
                table = batch.filter(pc.is_in(batch.column(1), value_set=wanted))
                if table.num_rows == 0:
                    continue
                table = table.filter(pc.is_in(table.column(0), value_set=allowed))
                if table.num_rows == 0:
                    continue
                d = table.to_pandas()
                d['datetime'] = pd.to_datetime(d.datetime, errors='coerce')
                d['entertime'] = pd.to_datetime(d.entertime, errors='coerce')
                d['patientid'] = d.patientid.astype('int64')
                a = d.loc[d.variableid.eq(15001552)].copy()
                if len(a):
                    airway.append(a)
                u = d.loc[~d.variableid.eq(15001552)].copy()
                if u.empty:
                    continue
                u['admissiontime'] = u.patientid.map(ref_admission)
                u['hrs_measured'] = key_time(u, 'datetime', 'admissiontime')
                u['hrs_entered'] = key_time(u, 'entertime', 'admissiontime')
                u = u.loc[(u.hrs_measured >= 0) & (u.hrs_measured <= 2) & (u.hrs_entered >= 0) & (u.hrs_entered <= 2)].copy()
                if len(u):
                    kept.append(u)
                    sel_count += len(u)
            if parts % 50 == 0:
                (OUT / 'checkpoint.json').write_text(json.dumps({'parts': parts, 'features_kept': sel_count}), encoding='utf-8')
                print('CLEAN_SCAN ' + str(parts) + ' parquet partitions, kept ' + str(sel_count) + ' records', flush=True)
    REPORT['parquet_parts_scanned'] = parts
    REPORT['clean_raw_observations'] = sel_count
    if parts != 250 or not kept or (not airway):
        raise RuntimeError('Not all raw observation partitions or airway records available')
    obs = pd.concat(kept, ignore_index=True)
    aw = pd.concat(airway, ignore_index=True)
    aw['admissiontime'] = aw.patientid.map(ref_admission)
    aw['hours_measured'] = key_time(aw, 'datetime', 'admissiontime')
    aw['hours_entered'] = key_time(aw, 'entertime', 'admissiontime')
    aw = aw.dropna(subset=['datetime']).sort_values(['patientid', 'datetime', 'entertime'], kind='mergesort')
    decisions = []
    for pid, g in aw.groupby('patientid', sort=False):
        g = g.loc[g.value.isin([1, 2, 3, 4, 5])].copy()
        first_niv = g.loc[g.value.isin([3, 4, 5])].head(1)
        if first_niv.empty:
            decisions.append({'patientid': pid, 'reason': 'NO_RAW_NIV', 'first_niv_h': np.nan, 'event_h': np.nan})
            continue
        onset = first_niv.iloc[0]
        later = g.loc[g.datetime > onset.datetime]
        invasive = later.loc[later.value.isin([1, 2])].head(1)
        evt = invasive.iloc[0] if len(invasive) else None
        decisions.append({'patientid': int(pid), 'reason': 'OK', 'first_niv_h': float(onset.hours_measured), 'first_niv_entry_h': float(onset.hours_entered), 'event_h': float(evt.hours_measured) if evt is not None else np.nan, 'event_entry_h': float(evt.hours_entered) if evt is not None else np.nan})
    ev = pd.DataFrame(decisions).set_index('patientid')
    idx = pd.Index(sorted(ids), name='patientid')
    ev = ev.reindex(idx)
    active = ev.first_niv_h.ge(0) & ev.first_niv_h.le(2) & ev.first_niv_entry_h.le(2) & (ev.event_h.isna() | ev.event_h.gt(2))
    active = active.fillna(False)
    event_after = ev.event_h.gt(2) & active
    reported = eligible.set_index('patientid')
    prior_transition = reported['change group'].eq('Invasive Failure').reindex(idx).fillna(False)
    discordant = (prior_transition != ev.event_h.notna()).fillna(True)
    clean_ids = idx[active & ~discordant]
    REPORT.update({'first_niv_present': int(ev.first_niv_h.notna().sum()), 'first_niv_by_2h': int((ev.first_niv_h.le(2) & ev.first_niv_entry_h.le(2)).sum()), 'raw_intubation_events_anytime': int(ev.event_h.notna().sum()), 'raw_intubation_at_or_before_2h': int(ev.event_h.le(2).sum()), 'source_vs_raw_outcome_discordant': int(discordant.sum()), 'risk_set_before_discordant_exclusions': int(active.sum()), 'risk_set_after_exclusions': len(clean_ids), 'post_landmark_intubations': int(event_after.reindex(clean_ids).sum())})
    if len(clean_ids) < 900 or event_after.reindex(clean_ids).sum() < 250:
        raise RuntimeError('Unexpectedly small clean risk set or raw outcome coverage')
    obs = obs.loc[obs.patientid.isin(clean_ids)].copy()
    obs['variable_name'] = obs.variableid.map(variable_name)
    obs = obs.dropna(subset=['value', 'variable_name']).sort_values(['patientid', 'variable_name', 'datetime', 'entertime'], kind='mergesort')
    grp = obs.groupby(['patientid', 'variable_name'], sort=False)['value'].agg(['first', 'last', 'mean'])
    melted = {suf: grp[suf].unstack('variable_name').rename(columns=lambda c: c + '_' + suf) for suf in ('first', 'last', 'mean')}
    clean = pd.DataFrame(index=clean_ids)
    for name in feature_columns:
        kind = name.rsplit('_', 1)[1]
        frame = melted[kind]
        clean[name] = frame[name].reindex(clean_ids) if name in frame else np.nan
    clean['age'] = pd.to_numeric(ref.age.reindex(clean_ids), errors='coerce')
    clean['sex'] = ref.sex.reindex(clean_ids).astype(str).map({'M': 1.0, 'F': 0.0, 'Male': 1.0, 'Female': 0.0})
    if clean.sex.isna().mean() > 0.01:
        raise RuntimeError('Unknown reference sex coding')
    clean['outcome_intubation'] = event_after.reindex(clean_ids).astype(int).to_numpy()
    clean['outcome_composite_provisional'] = (event_after.reindex(clean_ids).to_numpy() | ref.discharge_status.reindex(clean_ids).eq('dead').to_numpy()).astype(int)
    clean['patientid'] = clean.index.astype(int)
    clean = clean.reset_index(drop=True)
    clean.to_parquet(OUT / 'PRIVATE_CLEAN_STRICT_2H_MODEL_MATRIX.parquet', index=False)
    ev.loc[clean_ids].to_csv(OUT / 'PRIVATE_RAW_AIRWAY_EVENT_MANIFEST.csv')
    (OUT / 'PRIVATE_SOURCE_EXCLUSIONS.csv').write_text(ev.assign(active=active, discordant=discordant).to_csv(), encoding='utf-8')
    feat_meta = []
    for c in ['age', 'sex'] + feature_columns:
        feat_meta.append({'name': c, 'availability': 'measured AND entered no later than ICU+2h' if c not in ('age', 'sex') else 'admission reference demographics', 'missing_n': int(clean[c].isna().sum()), 'missing_fraction': float(clean[c].isna().mean()), 'aggregation': c.rsplit('_', 1)[-1] if c not in ('age', 'sex') else 'admission'})
    (OUT / 'FEATURE_MANIFEST.json').write_text(json.dumps({'features': feat_meta, 'excluded_unvalidated_features': omitted, 'clock_definition': 'ICU admission + 2h; datetime and entertime both checked', 'not_the_original_claimed_28_feature_model': True}, indent=2), encoding='utf-8')
    REPORT.update({'status': 'CLEAN_COHORT_RECONSTRUCTED', 'clean_n': len(clean), 'clean_intubations': int(clean.outcome_intubation.sum()), 'composite_provisional_events': int(clean.outcome_composite_provisional.sum()), 'predictor_columns': len(feature_columns) + 2, 'predictor_missingness_ge_30pct': sum((q['missing_fraction'] > 0.3 for q in feat_meta)), 'zero_observation_features': sum((q['missing_n'] == len(clean) for q in feat_meta)), 'death_time_available': False, 'composite_time_verified': False, 'intubation_time_verified_for_positive_cases': True, 'raw_input_matrix_private': str(OUT / 'PRIVATE_CLEAN_STRICT_2H_MODEL_MATRIX.parquet')})
    save()
    return 0
if __name__ == '__main__':
    if not ROOT.is_dir():
        raise FileNotFoundError(f'Private research root is unavailable: {ROOT}. Set NIV_RESEARCH_ROOT.')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, OUT / 'source_snapshot.py')
    try:
        code = main()
    except Exception as e:
        REPORT['status'] = 'BUILD_FAILED'
        REPORT['error_type'] = type(e).__name__
        REPORT['error'] = str(e)[:250]
        (OUT / 'PRIVATE_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        save()
        code = 1
    sys.exit(code)
