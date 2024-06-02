import numpy as np
from pandas import DataFrame
from datetime import date, datetime
from tqdm import tqdm
import data_processing
import models


def rsf_rfc(df: DataFrame, censor_vals: list, time, rsf_params=None, rfc_params=None):
    datasets = data_processing.create_datasets(df, time)
    uncensored_train = datasets['uncensored']['X_train'].shape[0]
    test_size = datasets['test']['X'].shape[0]
    rsf_aucs = []
    rfc_aucs = []
    for i in tqdm(censor_vals):
        rfc_aucs.append(models.rfc_score(datasets['uncensored']['X_train'], datasets['uncensored']['y_train'],
                                   datasets['test']['X'], datasets['uncensored']['y_test'], rfc_params))

        X_train, y_train = data_processing.use_censored_samples(datasets['censored']['X_train'],
                                                                datasets['censored']['y_train'], i, time)
        # Make sure the test size is 20% of the total dataset when considering the ratio of censoring
        test_X_modified = datasets['test']['X']#[:int(min(datasets['test']['X'].shape[0], X_train.shape[0]*0.2*1.2))]
        test_y_modified = datasets['test']['y']#[:int(min(datasets['test']['X'].shape[0], X_train.shape[0]*0.2*1.2))]
        auc_rsf = (models.rsf_score(X_train, y_train, test_X_modified, test_y_modified, time, rsf_params)
                   .item())
        rsf_aucs.append(auc_rsf)
    return np.asarray(rfc_aucs).mean(), rsf_aucs, datasets


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
