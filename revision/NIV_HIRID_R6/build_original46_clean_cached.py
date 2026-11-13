"""Construct the locked 46-predictor matrix using the original ICU 0 to 2 hour window.

Outputs are stored locally with a schema and SHA256 manifest.
"""
import os, io, json, hashlib, tarfile, traceback, sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
ROOT = Path(os.environ.get('NIV_RESEARCH_ROOT', 'D:\\ML\\research_campaign'))
BASE = ROOT / 'datasets' / 'HiRID'
SRC = Path(os.getenv('NIV_ORIGINAL44_BUILD_DIR') or ROOT / 'runs' / 'NIV_R6_ORIGINAL_44_COLS_STRICT_RAW_BUILD' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1')))
RAW = BASE / 'raw_official_v1_1_1'
DEST = BASE / 'model_inputs'
NAME = 'NIV_original46_strict2h_v1'
CSV = DEST / (NAME + '.csv')
META = DEST / (NAME + '.manifest.json')
RUN = ROOT / 'runs' / 'NIV_R6_ORIGINAL46_ONCE' / (os.getenv('NIV_RUN_ID') or os.environ.get('GITHUB_RUN_ID', 'manual') + '-' + os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
COLS = ['patientid', 'sex', 'age', 'ventilation time', 'Outcome', 'Base excess in blood_first', 'Bicarbonate in blood_first', 'End tidal co2 _first', 'Glucose  in blood _first', 'Hg in Blood_first', 'Lactate in blood_first', 'Mean inspiratory airway pressure_first', 'Oxygen saturation in blood_first', 'Peripheral oxygen saturation_first', 'Potassium in blood_first', 'Richmond agitation-sedation scale_first', 'Sodium in blood_first', 'Ventilator mode_first', 'CO2 in blood_first', 'PH of blood_first', 'Peep setting_first', 'Base excess in blood_last', 'Bicarbonate in blood_last', 'End tidal co2 _last', 'Glucose  in blood _last', 'Hg in Blood_last', 'Lactate in blood_last', 'Mean inspiratory airway pressure_last', 'Oxygen saturation in blood_last', 'Peripheral oxygen saturation_last', 'Potassium in blood_last', 'Richmond agitation-sedation scale_last', 'Sodium in blood_last', 'Ventilator mode_last', 'CO2 in blood_last', 'PH of blood_last', 'Peep setting_last', 'Heart rate_mean', 'Mean arterial pressure_mean', 'Respiratory rate_mean', 'GCS_first', 'GCS_last', 'PO2 FiO2 ratio_first', 'PO2 FiO2 ratio_last', 'Cumulative Fluid balance_first', 'Cumulative Fluid balance_last', 'Lung complience_first', 'Lung complience_last']
PRED = [c for c in COLS if c not in ('patientid', 'Outcome')]
assert len(COLS) == 48 and len(PRED) == 46
GCS = {10000100: 'verbal', 10000200: 'motor', 10000300: 'eye'}
AIRWAY = 15001552

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def note(status, **info):
    private = dict(status=status, output_csv=str(CSV), run=str(RUN), **info)
    (RUN / 'summary.json').write_text(json.dumps(private, indent=2, default=str), encoding='utf8')
    print('NIV46 ' + json.dumps(dict(status=status, **info), default=str), flush=True)

def execute():
    if CSV.exists() or META.exists():
        if not CSV.is_file() or not META.is_file():
            raise ValueError('INCOMPLETE_EXISTING_CACHE_NO_OVERWRITE')
        manifest = json.loads(META.read_text(encoding='utf8'))
        if sha(CSV) != manifest['sha256'] or manifest['columns'] != COLS:
            raise ValueError('CACHE_HASH_OR_SCHEMA_MISMATCH_NO_OVERWRITE')
        note('VERIFIED_EXISTING_CSV_REUSED', rows=manifest['rows'], sha256=manifest['sha256'])
        return
    source = SRC / 'PRIVATE_CLEAN_STRICT_2H_MODEL_MATRIX.parquet'
    if not source.is_file():
        raise FileNotFoundError('CLEAN44_SOURCE_NOT_FOUND:' + str(source))
    clean = pd.read_parquet(source)
    if clean.patientid.duplicated().any():
        raise ValueError('DUPLICATE_PATIENTS')
    old_hash = sha(source)
    rename = {'co2 in blood_first': 'CO2 in blood_first', 'co2 in blood_last': 'CO2 in blood_last', 'pH of blood_first': 'PH of blood_first', 'pH of blood_last': 'PH of blood_last', 'peep setting_first': 'Peep setting_first', 'peep setting_last': 'Peep setting_last'}
    clean = clean.rename(columns=rename)
    absent = [c for c in PRED if c not in clean and c not in ('GCS_first', 'GCS_last', 'ventilation time')]
    if absent:
        raise ValueError('MISSING_EXISTING_FEATURES:' + str(absent))
    ids = set(pd.to_numeric(clean.patientid, errors='raise').astype('int64'))
    with tarfile.open(RAW / 'reference_data.tar.gz', 'r:gz') as tar:
        gen = pd.read_csv(io.BytesIO(tar.extractfile('general_table.csv').read()), usecols=['patientid', 'admissiontime'])
    gen.patientid = pd.to_numeric(gen.patientid, errors='raise').astype('int64')
    gen.admissiontime = pd.to_datetime(gen.admissiontime, errors='raise')
    if gen.patientid.duplicated().any():
        raise ValueError('DUPLICATE_ADMISSION_REFERENCES')
    admit = gen.set_index('patientid').admissiontime
    if not ids.issubset(set(admit.index)):
        raise ValueError('ADMISSION_TIMESTAMP_MISSING')
    allowed = pa.array(sorted(ids), type=pa.int32())
    vars = pa.array(sorted([*GCS, AIRWAY]), type=pa.int32())
    blocks = []
    part_count = 0
    with tarfile.open(RAW / 'observation_tables_parquet.tar.gz', 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith('.parquet'):
                continue
            part_count += 1
            handle = tar.extractfile(member)
            if handle is None:
                raise IOError('PARQUET_MEMBER_UNREADABLE')
            parquet = pq.ParquetFile(pa.BufferReader(handle.read()))
            for batch in parquet.iter_batches(batch_size=150000, columns=['patientid', 'variableid', 'value', 'datetime', 'entertime']):
                tab = pa.Table.from_batches([batch])
                tab = tab.filter(pc.is_in(tab.column('variableid'), value_set=vars))
                tab = tab.filter(pc.is_in(tab.column('patientid'), value_set=allowed))
                if tab.num_rows == 0:
                    continue
                a = tab.to_pandas()
                a['datetime'] = pd.to_datetime(a.datetime, errors='coerce')
                a['entertime'] = pd.to_datetime(a.entertime, errors='coerce')
                a['admission'] = a.patientid.map(admit)
                h_measured = (a.datetime - a.admission).dt.total_seconds() / 3600
                h_entered = (a.entertime - a.admission).dt.total_seconds() / 3600
                keep = a.loc[h_measured.between(0, 2) & h_entered.between(0, 2)].copy()
                if not keep.empty:
                    keep['hours_entered'] = h_entered.loc[keep.index].to_numpy()
                    blocks.append(keep)
            if part_count % 50 == 0:
                print('NIV46_SCANNED ' + str(part_count), flush=True)
    if part_count != 250 or not blocks:
        raise RuntimeError('INCOMPLETE_OFFICIAL_RAW_SCAN')
    obs = pd.concat(blocks, ignore_index=True).dropna(subset=['value'])
    obs = obs.sort_values(['patientid', 'variableid', 'datetime', 'entertime'], kind='mergesort')
    g = obs.loc[obs.variableid.isin(GCS)].copy()
    g['component'] = g.variableid.map(GCS)
    g['value'] = pd.to_numeric(g.value, errors='coerce')
    agg = g.groupby(['patientid', 'component']).value.agg(['first', 'last'])
    names = ['verbal', 'motor', 'eye']
    first = agg['first'].unstack('component').reindex(columns=names).sum(axis=1, min_count=3)
    last = agg['last'].unstack('component').reindex(columns=names).sum(axis=1, min_count=3)
    v = obs.loc[obs.variableid.eq(AIRWAY) & obs.value.isin([3, 4, 5])]
    init_hours = v.sort_values(['patientid', 'entertime', 'datetime'], kind='mergesort').groupby('patientid').first()['hours_entered']
    vent = np.floor(init_hours).astype(float)
    frame = clean.set_index('patientid', drop=False).copy()
    frame['GCS_first'] = frame.index.map(first)
    frame['GCS_last'] = frame.index.map(last)
    frame['ventilation time'] = frame.index.map(vent)
    if frame['ventilation time'].isna().any():
        raise ValueError('SOME_PATIENTS_HAVE_NO_VERIFIED_FIRST_2H_NIV')
    historical = BASE / 'drive_shared_sources' / '0_2 merged_respiratory_modelling final draft.csv'
    if not historical.is_file():
        raise FileNotFoundError('HISTORICAL_LABEL_CSV_MISSING')
    label_csv = pd.read_csv(historical, usecols=['patientid', 'Outcome'])
    if label_csv.patientid.duplicated().any():
        raise ValueError('DUPLICATE_LABELS')
    y = label_csv.set_index('patientid').Outcome
    if not ids.issubset(set(y.index)) or not y.isin(['Yes', 'No']).all():
        raise ValueError('LABELS_INCOMPLETE')
    frame['Outcome'] = frame.index.map(y)
    out = frame[COLS]
    if out.patientid.duplicated().any():
        raise ValueError('DUPLICATE_OUTPUT_PATIENTS')
    available = {k: int(out[k].notna().sum()) for k in ['GCS_first', 'GCS_last', 'ventilation time']}
    temp = DEST / (NAME + '.csv.partial.' + os.getenv('GITHUB_RUN_ID', 'manual'))
    tempmeta = DEST / (NAME + '.manifest.partial.' + os.getenv('GITHUB_RUN_ID', 'manual'))
    if temp.exists() or tempmeta.exists():
        raise ValueError('STALE_PARTIAL_EVIDENCE_EXISTS')
    out.to_csv(temp, index=False)
    hash_value = sha(temp)
    info = dict(columns=COLS, predictors=PRED, rows=len(out), sha256=hash_value, source_clean44_sha256=old_hash, source_clean44_path=str(source), raw_partitions=part_count, completeness=available, clock_rule='0<=datetime and entertime<=ICU_admission+2h', GCS_rule='sum of separately aggregated eye,verbal,motor within strict 2h', ventilation_rule='floor(first qualifying NIV entry - admission in hours)', label_source='historical final draft CSV', limitations='Exact historical row values need audit; death times unverified', immutable=True, no_model_training=True)
    tempmeta.write_text(json.dumps(info, indent=2), encoding='utf8')
    temp.rename(CSV)
    tempmeta.rename(META)
    note('CANONICAL_CSV_SAVED', rows=len(out), sha256=hash_value, completeness=available)
if __name__ == '__main__':
    if not ROOT.is_dir():
        raise FileNotFoundError(f'Private research root is unavailable: {ROOT}. Set NIV_RESEARCH_ROOT.')
    DEST.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=False)
    try:
        execute()
    except Exception as error:
        (RUN / 'PRIVATE_traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
        note('BLOCKED_NO_OVERWRITE', error=type(error).__name__ + ':' + str(error)[:180])
        sys.exit(1)
