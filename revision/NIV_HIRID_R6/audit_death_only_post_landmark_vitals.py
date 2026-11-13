"""Assess post-landmark observation evidence for deaths without an intubation event.

HiRID 1.1.1 records mortality status but does not provide a verified death
timestamp in the reference table. Subsequent plausible clinical observations
support a lower bound on survival, not an estimate of time of death.
"""
import io, os, sys, json, shutil, tarfile, traceback, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
ROOT = Path(os.environ.get('NIV_RESEARCH_ROOT', 'D:\\ML\\research_campaign'))
RAW = ROOT / 'datasets' / 'HiRID' / 'raw_official_v1_1_1'
SRC = ROOT / 'datasets' / 'HiRID' / 'drive_shared_sources'
BUILT = Path(os.getenv('NIV_CLEAN_BUILD_DIR') or ROOT / 'runs' / 'NIV_R6_CLEAN_LANDMARK_BUILD' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1')))
OUT = ROOT / 'runs' / 'NIV_R6_DEATH_EVENT_LOWER_BOUND_AUDIT' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1'))
START = time.monotonic()
report = {'status': 'STARTED', 'hypothesis': 'Post-death clinical measurement is not expected; test whether deaths have independent evidence of being alive after two hour landmark', 'no_death_timestamp_in_reference': True, 'important': 'Last observation time cannot be used as recorded death time'}

def save():
    (OUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('DEATH_TIMING_AUDIT ' + json.dumps(report, default=str), flush=True)
if __name__ == '__main__':
    if not ROOT.is_dir():
        raise FileNotFoundError(f'Private research root is unavailable: {ROOT}. Set NIV_RESEARCH_ROOT.')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, OUT / 'source_snapshot.py')
    try:
        with tarfile.open(RAW / 'reference_data.tar.gz', 'r:gz') as tar:
            g = pd.read_csv(io.BytesIO(tar.extractfile('general_table.csv').read()), usecols=['patientid', 'admissiontime', 'discharge_status'])
        g['admissiontime'] = pd.to_datetime(g.admissiontime, errors='coerce')
        general = g.set_index('patientid')
        orig = pd.read_csv(SRC / '0_2 merged_respiratory_var_names_data 01 03 2024.csv', usecols=['patientid', 'ventilation type', 'change group'])
        old = pd.read_csv(SRC / '0_2 merged_respiratory_modelling final draft.csv', usecols=['patientid', 'Outcome'])
        clean = pd.read_parquet(BUILT / 'PRIVATE_CLEAN_STRICT_2H_MODEL_MATRIX.parquet', columns=['patientid', 'outcome_intubation', 'outcome_composite_provisional'])
        eligible = orig.loc[orig.patientid.isin(set(old.patientid)) & orig['ventilation type'].eq('Non-Invasive')].copy()
        eligible['dead'] = eligible.patientid.map(general.discharge_status).eq('dead')
        eligible['death_only'] = eligible.dead & ~eligible['change group'].eq('Invasive Failure')
        clean = clean.set_index('patientid')
        eligible['in_clean_cohort'] = eligible.patientid.isin(clean.index)
        eligible['clean_death_only'] = eligible.patientid.isin(clean.index) & eligible.dead & eligible.patientid.map(clean.outcome_intubation).eq(0)
        death = set(eligible.loc[eligible.dead, 'patientid'].astype(int))
        selected = set(eligible.patientid.astype(int))
        relevant_ids = {200: (30, 250), 110: (20, 150), 610: (20, 150), 4000: (40, 100), 8280: (40, 100), 300: (4, 70), 310: (4, 70), 20000300: (6.7, 7.8)}
        ids = pa.array(sorted(selected), type=pa.int32())
        codes = pa.array(sorted(relevant_ids), type=pa.int32())
        parts = 0
        rows = 0
        valid = []
        with tarfile.open(RAW / 'observation_tables_parquet.tar.gz', 'r|gz') as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith('.parquet'):
                    continue
                parts += 1
                pf = pq.ParquetFile(pa.BufferReader(tar.extractfile(m).read()))
                for batch in pf.iter_batches(batch_size=185000, columns=['patientid', 'variableid', 'datetime', 'entertime', 'value', 'status']):
                    t = batch.filter(pc.is_in(batch.column(1), value_set=codes))
                    if not t.num_rows:
                        continue
                    t = t.filter(pc.is_in(t.column(0), value_set=ids))
                    if not t.num_rows:
                        continue
                    d = t.to_pandas()
                    d['admissiontime'] = d.patientid.map(general.admissiontime)
                    d['datetime'] = pd.to_datetime(d.datetime, errors='coerce')
                    d['entertime'] = pd.to_datetime(d.entertime, errors='coerce')
                    d['measure_h'] = (d.datetime - d.admissiontime).dt.total_seconds() / 3600
                    d['entry_h'] = (d.entertime - d.admissiontime).dt.total_seconds() / 3600
                    lower = d.variableid.map({a: b[0] for a, b in relevant_ids.items()})
                    upper = d.variableid.map({a: b[1] for a, b in relevant_ids.items()})
                    d = d.loc[d.value.between(lower, upper) & d.measure_h.ge(0) & d.entry_h.ge(0)].copy()
                    if d.empty:
                        continue
                    rows += len(d)
                    valid.append(d.loc[:, ['patientid', 'variableid', 'measure_h', 'entry_h', 'status']])
                if parts % 50 == 0:
                    print('DEATH_SCAN', parts, rows, flush=True)
        if parts != 250 or not valid:
            raise RuntimeError('Incomplete death landmark observation audit')
        v = pd.concat(valid, ignore_index=True)
        v['after2'] = (v.measure_h > 2) & (v.entry_h > 2)
        v['after3'] = (v.measure_h > 3) & (v.entry_h > 3)
        v['after6'] = (v.measure_h > 6) & (v.entry_h > 6)
        group = v.groupby('patientid', sort=False)
        patient = eligible.set_index('patientid')[['dead', 'death_only', 'in_clean_cohort', 'clean_death_only']].copy()
        for key, func in {'vital_any_at_0_to_2': lambda x: (x.measure_h.between(0, 2) & x.entry_h.between(0, 2)).any(), 'vital_after_2h': lambda x: x.after2.any(), 'vital_after_3h': lambda x: x.after3.any(), 'vital_after_6h': lambda x: x.after6.any()}.items():
            patient[key] = group.apply(func).reindex(patient.index).fillna(False).astype(bool)
        patient['max_measure_h'] = group.measure_h.max().reindex(patient.index)
        patient['max_enter_h'] = group.entry_h.max().reindex(patient.index)
        patient['vital_n'] = group.size().reindex(patient.index).fillna(0).astype(int)
        patient.to_csv(OUT / 'PRIVATE_DEATH_ONLY_VITAL_TIME_BOUND.csv')
        summary = {}
        for label, mask in [('eligible_all', np.ones(len(patient), dtype=bool)), ('any_icu_death', patient.dead), ('historical_death_only', patient.death_only), ('death_only_in_new_clean_cohort', patient.clean_death_only), ('deaths_in_clean_cohort', patient.dead & patient.in_clean_cohort)]:
            tab = patient.loc[mask]
            summary[label] = {'n': len(tab), 'with_valid_vitals_0_to_2': int(tab.vital_any_at_0_to_2.sum()), 'with_vitals_after_2h': int(tab.vital_after_2h.sum()), 'with_vitals_after_3h': int(tab.vital_after_3h.sum()), 'with_vitals_after_6h': int(tab.vital_after_6h.sum()), 'without_post2h_vital_evidence': int((~tab.vital_after_2h).sum())}
        report.update({'status': 'DEATH_VITAL_LOWER_BOUND_AUDIT_COMPLETED', 'source_partitions': parts, 'rows_with_physically_plausible_vitals': rows, 'cohort_summary': summary, 'conclusion_rule': 'Only plausible clinical vitals measured AND entered after ICU+2h support survival past that time; absence does not locate death; evidence may include automated or retrospective charting.', 'death_time_verified': False, 'elapsed_s': round(time.monotonic() - START, 1), 'private_dir': str(OUT)})
        save()
    except Exception as e:
        report.update({'status': 'FAILED', 'error_type': type(e).__name__, 'error': str(e)[:200]})
        (OUT / 'PRIVATE_TRACEBACK.txt').write_text(traceback.format_exc(), encoding='utf-8')
        save()
        sys.exit(1)
