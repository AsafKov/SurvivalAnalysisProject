import numpy as np
from pandas import DataFrame
from datetime import date, datetime
import data_processing
import models


def run_experiment(df: DataFrame, censor_vals: list, censor_time, exp_models, models_params, test_size_relative=True, debug=False):
    surv_model, sk_model = exp_models
    surv_params, sk_params = models_params
    datasets = data_processing.create_datasets(df, censor_time, test_size_relative)
    survival_model_aucs = []
    sklearn_avg_aug = 0
    if debug:
        uncensored_train_size = datasets['uncensored']['X_train'].shape[0]
        test_size = datasets['test']['X'].shape[0]
        print(f'uncensored train size: {uncensored_train_size}, test size: {test_size}')

    for i in censor_vals:
        model = models.sklearn_model_fit(datasets['uncensored']['X_train'], datasets['uncensored']['y_train'], sk_model, sk_params)
        sklearn_avg_aug += models.sklearn_model_score(model, datasets['test']['X'], datasets['uncensored']['y_test'])
        X_train, y_train = data_processing.use_censored_samples(datasets['censored']['X_train'],
                                                                datasets['censored']['y_train'], i, censor_time)
        if debug:
            train_size = X_train.shape[0]
            print(f'censored train size ({i*100}% censoring): {train_size}')
        model = models.survival_model_fit(X_train, y_train, surv_model, surv_params)
        auc_rsf = models.survival_model_score(model,  datasets['test']['X'], datasets['test']['y'], y_train, censor_time).item()
        survival_model_aucs.append(auc_rsf)
    return sklearn_avg_aug/len(censor_vals), survival_model_aucs, datasets


# saves the experiment and its results in a text file in the results folder
def exp_report(df_size, name, censor_vals, datasets, time, rsf_params, rfc_params, results):
    today = date.today()
    time_in_day = datetime.now()
    test_size = datasets['test']['X'].shape[0]
    uncensored_train_sizes = datasets['uncensored']['X_train'].shape[0]
    n_features = datasets['test']['X'].shape[1]
    censored_train_sizes = []
    for i in censor_vals:
        X_train, y_train = data_processing.use_censored_samples(datasets['censored']['X_train'],
                                                                datasets['censored']['y_train'], i, time)
        censored_train_sizes.append(X_train.shape[0])
    file = open(f"results/{name}_{today.strftime('%d-%m-%Y')}-{time_in_day.strftime('%H-%M-%S')}.txt", 'a')
    file.write(f"{name} dataset, {df_size} samples, censored: "
               f"{censor_vals[-1]*100:.2f}%\n\n")
    file.write(f"RFS train censored percentages: {np.asarray(censor_vals)*100}\n")
    file.write(f'features: {n_features}\n')
    file.write(f'test size: {test_size}\n')
    file.write(f'uncensored train size: {uncensored_train_sizes}\n')
    file.write(f'censored train size: {censored_train_sizes}\n')
    file.write(f"RSF parameters: {rsf_params}\n")
    file.write(f"RFC parameters: {rfc_params}\n")
    file.write(f"Results: {results}")
    file.close()
