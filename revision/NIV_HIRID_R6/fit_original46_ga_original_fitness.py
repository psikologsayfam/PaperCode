"""Nested evaluation of six base learners and genetic ensemble weights.

Input features are reconstructed within the first two hours after ICU admission.
The retained historical composite endpoint lacks exact mortality timestamps;
three death only cases without post-landmark survival evidence are excluded.
This retrospective analysis is exploratory rather than independently validated.
"""
from pathlib import Path
import os, json, sys, shutil, traceback, time, hashlib
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
SEED = 20261027
ROOT = Path(os.environ.get('NIV_RESEARCH_ROOT', 'D:\\ML\\research_campaign'))
BUILD = Path(os.getenv('NIV_ORIGINAL44_BUILD_DIR') or ROOT / 'runs' / 'NIV_R6_ORIGINAL_44_COLS_STRICT_RAW_BUILD' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1')))
CACHE = ROOT / 'datasets' / 'HiRID' / 'model_inputs' / 'NIV_original46_strict2h_v1.csv'
CACHE_META = CACHE.with_suffix('.manifest.json')
OUT = ROOT / 'runs' / 'NIV_R6_ORIGINAL46_STRICT2H_GA_ORIGINAL_FITNESS' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1'))
models = ['DT', 'RF', 'GB', 'KNN', 'LGBM', 'XGB']
T0 = time.monotonic()
report = {'status': 'RUNNING', 'experiment': 'Exact original 46 CSV predictor columns with strict 0 to 2 hour provenance; six models, fold-only KNN impute, train-only SMOTE, original fitness, search budget 60x80', 'run_id': os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1'), 'selection': 'Fixed original GA 0.5*(AUROC + F1 at 0.5) on honest training-only inner OOF predictions', 'imputation': 'KNNImputer n_neighbors=5 inside training CV', 'scaling': 'StandardScaler inside training CV', 'oversampling': 'SMOTE k_neighbors=3 fit only on training split, never on validation/test', 'seed': SEED, 'GA_population': 60, 'GA_generations': 80, 'base_models': models, 'external_comparator': 'Logistic Regression', 'study_design': 'Exploratory retrospective sensitivity analysis; independent validation unavailable'}

def emit():
    (OUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('ORIGINAL46_GA_SUMMARY ' + json.dumps(report, default=str), flush=True)

def pipeline(name, seed):
    steps = [('imputer', KNNImputer(n_neighbors=5)), ('scale', StandardScaler()), ('smote', SMOTE(random_state=seed, k_neighbors=3))]
    if name == 'DT':
        est = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=seed)
    elif name == 'RF':
        est = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=3, n_jobs=2, random_state=seed)
    elif name == 'GB':
        est = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.06, random_state=seed)
    elif name == 'KNN':
        est = KNeighborsClassifier(n_neighbors=11)
    elif name == 'LGBM':
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(n_estimators=140, learning_rate=0.05, num_leaves=15, min_child_samples=20, n_jobs=2, verbosity=-1, random_state=seed)
    elif name == 'XGB':
        from xgboost import XGBClassifier
        est = XGBClassifier(n_estimators=140, learning_rate=0.05, max_depth=3, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', objective='binary:logistic', n_jobs=2, random_state=seed)
    elif name == 'LR':
        est = LogisticRegression(C=1.0, max_iter=1500, random_state=seed)
    else:
        raise ValueError(name)
    steps.append(('model', est))
    return Pipeline(steps)

def grid(name):
    return {'DT': {'model__max_depth': [3, 6], 'model__min_samples_leaf': [5, 15]}, 'RF': {'model__min_samples_leaf': [2, 6], 'model__max_depth': [6, None]}, 'GB': {'model__n_estimators': [80, 160], 'model__learning_rate': [0.04, 0.1]}, 'KNN': {'model__n_neighbors': [5, 11, 19]}, 'LGBM': {'model__num_leaves': [7, 15, 31]}, 'XGB': {'model__max_depth': [2, 4]}, 'LR': {'model__C': [0.1, 1.0, 10.0]}}[name]

def tune(name, x, y, seed):
    cv = StratifiedKFold(2, shuffle=True, random_state=seed + 13)
    gs = GridSearchCV(pipeline(name, seed), grid(name), cv=cv, scoring='f1', n_jobs=1, refit=True, error_score='raise')
    gs.fit(x, y)
    return (gs.best_estimator_, gs.best_params_)

def ga_weights(pred, y, seed, pop=60, gens=80):
    """Optimise the equal-weight AUROC and F1 objective using inner out of fold predictions."""
    rng = np.random.default_rng(seed)
    n_model = pred.shape[1]
    population = rng.dirichlet(np.ones(n_model), size=pop)
    history = []

    def fitness(w):
        pp = np.clip(pred @ w, 0, 1)
        return (roc_auc_score(y, pp) + f1_score(y, pp >= 0.5, zero_division=0)) * 0.5
    best = None
    score = -1
    for gen in range(gens):
        scores = np.array([fitness(w) for w in population])
        elite = population[np.argsort(scores)[-6:]]
        j = int(np.argmax(scores))
        if scores[j] > score:
            score = float(scores[j])
            best = population[j].copy()
        history.append({'generation': gen + 1, 'inner_objective': score})
        new = [w.copy() for w in elite]
        while len(new) < pop:
            ids = rng.choice(pop, 3, replace=False)
            a = population[ids[np.argmax(scores[ids])]]
            ids = rng.choice(pop, 3, replace=False)
            b = population[ids[np.argmax(scores[ids])]]
            alpha = rng.random()
            child = alpha * a + (1 - alpha) * b
            if rng.random() < 0.25:
                child = np.maximum(0, child + rng.normal(0, 0.06, size=n_model))
            if np.sum(child) == 0:
                child = np.ones(n_model)
            new.append(child / child.sum())
        population = np.array(new)
    return (best, history)

def delong(y, a, b):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    arr = []
    for p in (a, b):
        comp = p[pos, None] - p[None, neg]
        h = (comp > 0).astype(float) + 0.5 * (comp == 0)
        arr.append([h.mean(axis=1), h.mean(axis=0)])
    v10 = np.array([a[0] for a in arr])
    v01 = np.array([a[1] for a in arr])
    var = np.array([1, -1]) @ (np.cov(v10) / len(pos) + np.cov(v01) / len(neg)) @ np.array([1, -1])
    delta = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    if var <= 0:
        return {'delta': delta, 'p_two_sided': None, 'CI95': None}
    se = float(np.sqrt(var))
    return {'delta': delta, 'CI95': [delta - 1.96 * se, delta + 1.96 * se], 'p_two_sided': float(2 * norm.sf(abs(delta / se)))}

def fresh_bootstrap(y, pred, nboot=1200):
    rng = np.random.default_rng(SEED + 991)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    aucs = {k: [] for k in pred}
    dif = []
    for _ in range(nboot):
        z = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        a = {k: roc_auc_score(y[z], p[z]) for k, p in pred.items()}
        for k in aucs:
            aucs[k].append(a[k])
        dif.append(a['GA_composite'] - a['Uniform'])
    return {'AUROC_CI95': {k: list(map(float, np.quantile(v, [0.025, 0.975]))) for k, v in aucs.items()}, 'GA_minus_Uniform_CI95': list(map(float, np.quantile(dif, [0.025, 0.975])))}

def main():
    mortality = Path(os.getenv('NIV_DEATH_AUDIT_DIR') or ROOT / 'runs' / 'NIV_R6_DEATH_EVENT_LOWER_BOUND_AUDIT' / (os.getenv('NIV_RUN_ID') or os.getenv('GITHUB_RUN_ID', 'manual') + '-' + os.getenv('GITHUB_RUN_ATTEMPT', '1')))
    mort = json.loads((mortality / 'SUMMARY.json').read_text(encoding='utf-8'))
    if mort['status'] != 'DEATH_VITAL_LOWER_BOUND_AUDIT_COMPLETED':
        raise RuntimeError('Death-only event lower bound not confirmed')
    evidence = pd.read_csv(mortality / 'PRIVATE_DEATH_ONLY_VITAL_TIME_BOUND.csv', index_col=0)
    for c in ('clean_death_only', 'vital_after_2h'):
        evidence[c] = evidence[c].astype(str).str.lower().eq('true')
    deathonly = evidence.loc[evidence.clean_death_only]
    if len(deathonly) != 39 or deathonly.vital_after_2h.sum() != 36:
        raise RuntimeError('Expected 39 death only, 36 with landmark survival evidence')
    unresolved = set(map(int, deathonly.loc[~deathonly.vital_after_2h].index))
    if len(unresolved) != 3:
        raise RuntimeError('Need 3 missing death-time cases excluded from this sensitivity')
    import ast
    if not CACHE.is_file() or not CACHE_META.is_file():
        raise FileNotFoundError('Permanent original46 CSV/manifest missing; do not regenerate automatically')
    info = json.loads(CACHE_META.read_text(encoding='utf8'))
    h = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    if h != info['sha256']:
        raise RuntimeError('Cached CSV SHA mismatch, abort training')
    builder = ast.parse(Path(__file__).with_name('build_original46_clean_cached.py').read_text(encoding='utf8'))
    exact = [ast.literal_eval(n.value) for n in builder.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'COLS' for t in n.targets))]
    if len(exact) != 1:
        raise RuntimeError('Exact original46 header declaration unavailable')
    d46 = pd.read_csv(CACHE)
    if d46.columns.tolist() != exact[0] or info['columns'] != exact[0]:
        raise RuntimeError('Original46 column header mismatch')
    if len(d46) != 1274 or d46.patientid.duplicated().any():
        raise RuntimeError('Cached strict 2h cohort invalid')
    features = [z for z in exact[0] if z not in ('patientid', 'Outcome')]
    if len(features) != 46 or 'BMI' in features:
        raise RuntimeError('Wrong model feature list')
    if not d46.Outcome.isin(['Yes', 'No']).all():
        raise RuntimeError('Outcome encoding unexpected')
    mat = pd.read_parquet(BUILD / 'PRIVATE_CLEAN_STRICT_2H_MODEL_MATRIX.parquet')
    if len(mat) != 1274 or mat.patientid.duplicated().any():
        raise RuntimeError('Clean source matrix invalid')
    meta = mat[['patientid', 'outcome_composite_provisional', 'outcome_intubation']]
    d46 = d46.merge(meta, on='patientid', validate='one_to_one', how='left')
    if d46.outcome_intubation.isna().any():
        raise RuntimeError('Missing strict cohort event provenance')
    d = d46.loc[~d46.patientid.isin(unresolved)].copy().reset_index(drop=True)
    X = d[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    y = d.Outcome.map({'No': 0, 'Yes': 1}).to_numpy(dtype=int)
    observed_intub = d.outcome_intubation.astype(int).to_numpy()
    source_label_disagreement = int((y != d.outcome_composite_provisional.astype(int).to_numpy()).sum())
    if len(d) != 1271 or observed_intub.sum() != 454:
        raise RuntimeError('Conservative risk-set patients changed')
    if X.isna().all().any():
        raise RuntimeError('At least one original input column is entirely empty')
    report.update({'n': len(d), 'composite_events': int(y.sum()), 'postlandmark_intubations': int(observed_intub.sum()), 'postlandmark_survival_supported_death_only': 36, 'excluded_death_cases_without_survival_evidence': 3, 'feature_count': len(features), 'historical_actual_feature_columns': features, 'source_label_disagreement': source_label_disagreement, 'label_source': 'Original historical Outcome from cached CSV; mortality timestamps not fully verified', 'composite_death_time_exact': False, 'evidence_warning': '39 death-only cases have no precise death timestamp; three excluded by historical conservative rule; remaining estimates exploratory', 'cache_csv_sha256': h, 'cache_csv_path': str(CACHE)})
    (OUT / 'CONFIG.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 45)
    names = models + ['GA_composite', 'Uniform', 'LR']
    pred = {n: np.full(len(d), np.nan) for n in names}
    fold_ids = np.zeros(len(d), dtype=int)
    fold_rows = []
    with threadpool_limits(limits=2):
        for fold, (tr, te) in enumerate(cv.split(X, y), 1):
            print('COMPOSITE_OUTER_FOLD', fold, 'START elapsed', round(time.monotonic() - T0, 1), flush=True)
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED + 45 + fold)
            pp = np.full((len(tr), len(models)), np.nan)
            tuning = []
            for i, (itr, iv) in enumerate(inner.split(X.iloc[tr], y[tr]), 1):
                for k, name in enumerate(models):
                    model, param = tune(name, X.iloc[tr[itr]], y[tr[itr]], SEED + 45000 + 1000 * fold + 100 * i + k)
                    pp[iv, k] = model.predict_proba(X.iloc[tr[iv]])[:, 1]
                    tuning.append({'inner': i, 'name': name, 'params': param})
            if not np.isfinite(pp).all():
                raise RuntimeError('Inner OOF leakage or gaps')
            w, h = ga_weights(pp, y[tr], SEED + 55000 + fold, pop=60, gens=80)
            pp_test = np.zeros((len(te), len(models)))
            fold_dir = OUT / f'PRIVATE_FOLD_{fold}'
            fold_dir.mkdir(exist_ok=True)
            for k, name in enumerate(models):
                model, param = tune(name, X.iloc[tr], y[tr], SEED + 80000 + 1000 * fold + k)
                pp_test[:, k] = model.predict_proba(X.iloc[te])[:, 1]
                pred[name][te] = pp_test[:, k]
                tuning.append({'stage': 'outer_full_training', 'name': name, 'params': param})
                import joblib
                joblib.dump(model, fold_dir / (name + '.joblib'))
            pred['GA_composite'][te] = pp_test @ w
            pred['Uniform'][te] = pp_test.mean(axis=1)
            lr_clf, lr_param = tune('LR', X.iloc[tr], y[tr], SEED + 90000 + fold)
            pred['LR'][te] = lr_clf.predict_proba(X.iloc[te])[:, 1]
            tuning.append({'stage': 'outer_full_training', 'name': 'LR', 'params': lr_param})
            import joblib
            joblib.dump(lr_clf, fold_dir / 'LR_baseline.joblib')
            fold_ids[te] = fold
            (fold_dir / 'GENETIC_WEIGHT_SELECTION.json').write_text(json.dumps({'ga_objective': '0.5*(AUROC+F1_threshold_0.5)', 'population': 60, 'generations': 80, 'weights': w.tolist(), 'inner_genetic_trace': h, 'base_tuning': tuning}, indent=2), encoding='utf-8')
            row = {'fold': fold, 'n': len(te), 'events': int(y[te].sum())}
            for name in names:
                row[name + '_AUROC'] = float(roc_auc_score(y[te], pred[name][te]))
            fold_rows.append(row)
            (OUT / 'PRIVATE_PROGRESS.json').write_text(json.dumps({'completed': fold, 'fold_rows': fold_rows, 'elapsed': time.monotonic() - T0}, indent=2), encoding='utf-8')
            print('COMPOSITE_FOLD_DONE', fold, 'GA_AUROC', round(row['GA_composite_AUROC'], 5), flush=True)
    if any((not np.isfinite(z).all() for z in pred.values())) or (fold_ids == 0).any():
        raise RuntimeError('Unscored patient in outer test')
    outcome = {name: {'AUROC': float(roc_auc_score(y, p)), 'AUPRC': float(average_precision_score(y, p)), 'Brier': float(brier_score_loss(y, p)), 'logloss': float(log_loss(y, np.clip(p, 1e-07, 1 - 1e-07))), 'F1_threshold_half': float(f1_score(y, p >= 0.5, zero_division=0))} for name, p in pred.items()}
    intervals = fresh_bootstrap(y, pred)
    tests = {k: delong(y, pred['GA_composite'], pred[k]) for k in ['Uniform', 'LR', 'RF']}
    for threshold in (0.3, 0.5, 0.7):
        tn, fp, fn, tp = map(int, confusion_matrix(y, pred['GA_composite'] >= threshold, labels=[0, 1]).ravel())
        outcome['GA_composite']['confusion_' + str(threshold)] = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    full = {'model_metrics': outcome, 'bootstrap': intervals, 'paired_Delong': tests, 'fold_rows': fold_rows, 'revised_population': report, 'source_death_survival_evidence': {'status': mort['status'], 'cohort_summary': mort['cohort_summary']}, 'caveat': 'Exact original 46 predictor CSV columns are reused from immutable strict 2h cache. Six model families, original combined F1/AUROC GA fitness, 60x80 generations and fold-only KNN imputation/SMOTE. This is an exploratory reanalysis with 3 death-only cases excluded. Original exact historical training hyperparameters and mortality event timestamps remain unverified.'}
    (OUT / 'PRIVATE_RESULTS.json').write_text(json.dumps(full, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame({'patientid': d.patientid, 'y_composite': y, 'y_intubation': observed_intub, 'outer_fold': fold_ids, **pred}).to_parquet(OUT / 'PRIVATE_COMPOSITE_OUTER_TEST_PREDICTIONS.parquet', index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / 'TABLE_COMPOSITE_OUTER_FOLDS.csv', index=False)
    pd.DataFrame([{'model': k, **v, 'AUROC_CI95_lower': intervals['AUROC_CI95'][k][0], 'AUROC_CI95_upper': intervals['AUROC_CI95'][k][1]} for k, v in outcome.items()]).to_csv(OUT / 'TABLE_COMPOSITE_MODEL_PERFORMANCE.csv', index=False)
    report.update({'status': 'ORIGINAL46_STRICT2H_GA_NESTED_COMPLETE', 'metrics': outcome, 'bootstrap': intervals, 'paired_Delong': tests, 'folds': fold_rows, 'private_dir': str(OUT), 'elapsed_s': round(time.monotonic() - T0, 1)})
    emit()
if __name__ == '__main__':
    if not ROOT.is_dir():
        raise FileNotFoundError(f'Private research root is unavailable: {ROOT}. Set NIV_RESEARCH_ROOT.')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, OUT / 'source_snapshot.py')
    try:
        main()
    except Exception as e:
        report.update({'status': 'FAILED', 'error_type': type(e).__name__, 'error': str(e)[:250]})
        (OUT / 'PRIVATE_TRACEBACK.txt').write_text(traceback.format_exc(), encoding='utf-8')
        emit()
        sys.exit(1)
