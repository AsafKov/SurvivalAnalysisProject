import numpy as np
from pandas import DataFrame
from datetime import date, datetime
from tqdm import tqdm
import data_processing
import models


def rsf_rfc(df: DataFrame, censor_vals: list, time, rsf_params=None, rfc_params=None):
    datasets = data_processing.create_datasets(df, time)

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
    return np.asarray(rfc_aucs).mean(), rsf_aucs


# saves the experiment and its results in a text file in the results folder
def exp_report(df, name, censored_vals, time, rsf_params, rfc_params, results):
    today = date.today()
    time_in_day = datetime.now()
    file = open(f"results/{name}_{today.strftime('%d-%m-%Y')}-{time_in_day.strftime('%H-%M-%S')}.txt", 'a')
    file.write(f"{name} dataset, {df.shape[0]} samples, censored: "
               f"{data_processing.censored_percentage(df, time):.2f}%\n\n")
    file.write(f"RSF parameters: {rsf_params}\n")
    file.write(f"RFC parameters: {rfc_params}\n")
    file.write(f"RFS train censored percentages: {np.asarray(censored_vals)*100}\n")
    file.write(f"Results: {results}")
    file.close()
