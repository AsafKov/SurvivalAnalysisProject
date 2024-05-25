import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import time

import data_processing


# train and test RSF model, output for a specific time (AUC)
def rsf_score(X_train, y_train, X_test, y_test, censor_time, rsf_params):
    rsf = RandomSurvivalForest(n_jobs=-1)
    if rsf_params != None:
        rsf.set_params(**rsf_params)
    y_train = y_structured_array(y_train)
    y_test = y_structured_array(y_test)
    rsf.fit(X_train, y_train)
    hazard_funcs_arr = rsf.predict(X_test)
    auc, _ = cumulative_dynamic_auc(survival_train=y_train, survival_test=y_test, estimate=hazard_funcs_arr,
                                    times=[censor_time])
    hazard_funcs_arr = rsf.predict(X_train)
    auc_t, _ = cumulative_dynamic_auc(survival_train=y_train, survival_test=y_train, estimate=hazard_funcs_arr,
                                    times=[censor_time])
    return auc


# train and test RF model, output for a specific time (AUC)
def rfc_score(X_train, y_train, X_test, y_test, rfc_params):
    rf = RandomForestClassifier(n_jobs=-1)
    if rfc_params != None:
        rf.set_params(**rfc_params)
    rf.fit(X_train, y_train)
    y_pred_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    return auc


# perform gridsearch cross-validation to find the best parameters for rfc
def model_tuning(X, y, model, params, verbose):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
    kf = KFold(n_splits=5, shuffle=False)
    gs_sv = GridSearchCV(estimator=model, param_grid=params, cv=kf, error_score="raise", verbose=verbose)
    gs_sv.fit(X_train, y_train)
    return gs_sv.best_params_, gs_sv.best_score_


# perform gridsearch cross-validation to find the best parameters for rsf
def model_tuning_rsf(df, censor_time, params, verbose=1):
    kf = KFold(n_splits=5, shuffle=True)
    best_auc = 0
    best_params = {}
    i = 0
    iterations = len(params['n_estimators'])*len(params['max_depth'])*len(params['min_samples_split'])*len(params['max_features'])
    datasets = data_processing.create_datasets(df, censor_time)
    X = datasets['censored']['X_train']
    y = datasets['censored']['y_train']
    start = time.time()
    for n in params['n_estimators']:
        for m in params['max_depth']:
            for l in params['min_samples_split']:
                for mf in params['max_features']:
                    i += 1
                    comb_scores = []
                    params_comb = {'n_estimators': n, 'max_depth': m, 'min_samples_split': l, 'max_features': mf}
                    c = 0
                    for train_index, test_index in kf.split(X, y):
                        c += 1
                        rsf = RandomSurvivalForest(n_estimators=n, max_depth=m, min_samples_leaf=l, max_features=mf, n_jobs=-1)
                        train_y = y_structured_array(y.iloc[train_index])
                        test_y = y_structured_array(y.iloc[test_index])
                        rsf.fit(X.iloc[train_index], train_y)
                        hazard_funcs_arr = rsf.predict(datasets['test']['X'])
                        auc, _ = cumulative_dynamic_auc(survival_train=train_y, survival_test=y_structured_array(datasets['test']['y']),
                                                        estimate=hazard_funcs_arr,
                                                        times=[censor_time])
                        comb_scores.append(auc.item())
                        if c == 3:
                            break
                    if np.asarray(comb_scores).mean() > best_auc:
                        best_auc = np.asarray(comb_scores).mean()
                        best_params = params_comb
                    iter_report = ''
                    if verbose >= 1:
                        iter_report += f"Iteration {i}/{iterations}"
                    if verbose >= 2:
                        iter_report += f", total time: {int(time.time()-start)}s, {int((time.time()-start)/i)}s/iter"
                    if verbose >= 3:
                        iter_report += f", score: {np.asarray(comb_scores).mean()}, params: {params_comb}"
                    print(iter_report)
    return best_params, best_auc


# yields structured-arrays with elements (censor_indicator, time) for both target sets
def y_structured_array(y):
    dt = np.dtype([('censor', np.bool_), ('time', np.int32)])
    y = list(y.itertuples(index=False, name=None))
    return np.array(y, dt)
