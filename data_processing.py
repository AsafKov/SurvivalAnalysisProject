from typing import Any
from sksurv.datasets import load_whas500, load_gbsg2, load_aids, load_flchain
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.metrics import eval_statistical


def generate_synthetic_data(original_data: DataFrame, num_samples: int, **kwargs: Any):
    rules = [(column, 'in', original_data[column].unique().tolist()) for column in original_data.select_dtypes(include=['category'])]
    rules = rules + [(column, 'dtype', original_data[column].dtype.name) for column in original_data.select_dtypes(exclude=['category'])]
    constraints = Constraints(rules=rules)
    syn_data_loader = SurvivalAnalysisDataLoader(
        original_data,
        target_column="censor",
        time_to_event_column="time",
        constrains=constraints
    )
    syn_gen = Plugins().get("survival_gan", **kwargs)
    syn_gen.fit(syn_data_loader)
    syn_data = syn_gen.generate(count=num_samples)
    df_syn = syn_data.raw()
    return syn_data_loader, df_syn


# Returns the censor statistics according to a given timeframe
def censored_percentage(syn_df: DataFrame, time: int):
    censored = (syn_df[(syn_df['time'] < time) & (syn_df['censor'] == False)].shape[0])/syn_df.shape[0]
    return censored*100


# returns the earliest timeframe with above 80% censoring (or as close as possible)
def determine_censor_time(df: DataFrame, target=70):
    time = 0
    top_percent = 0
    for i in range(int(df['time'].min()), int(df['time'].max()+1)):
        censored = censored_percentage(df, i)
        if censored > top_percent:
            top_percent = censored
            time = i
        if top_percent >= target:
            break
    return time, top_percent


# transform y of (censor, time) to a binary labels vector
def classification_transform(y: DataFrame, target_label="event"):
    y_classification = pd.DataFrame(columns=[target_label])
    y_classification[target_label] = np.logical_not(y['censor'])
    return y_classification


def data_evaluation(data1, data2):
    for col in data2.columns:
        data1[col] = data1[col].astype(int)
        data2[col] = data2[col].astype(int)
    eval_score = (f"Data evaluation:\nInverseKLDivergence:"
                  f" {eval_statistical.InverseKLDivergence().evaluate(data1, data2)}\n"
                  f"KolmogorovSmirnovTest: {eval_statistical.KolmogorovSmirnovTest().evaluate(data1, data2)}\n"
                  f"ChiSquaredTest: {eval_statistical.ChiSquaredTest().evaluate(data1, data2)}\n"
                  f"MaximumMeanDiscrepancy: {eval_statistical.MaximumMeanDiscrepancy().evaluate(data1, data2)}")
    return eval_score


def split_c_uc(df, time):
    df_censored = df[(df['time'] < time) & (df['censor'] == False)]
    df_uncensored = df[~((df['time'] < time) & (df['censor'] == False))]
    return df_censored, df_uncensored


# returns a dict of all the datasets required for an experiment, by a given timeframe
def create_datasets(df, time: int, relative_test_size=True):
    df_censored, df_uncensored = split_c_uc(df, time)
    X = df_uncensored.drop(labels=['censor', 'time'], axis=1)
    y = df_uncensored[['censor', 'time']]
    if relative_test_size:
        test_size = 0.55
    else:
        test_size = int(0.2*df_uncensored.shape[0])
    X_train_uc, X_test_uc, y_train_uc, y_test_uc = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_c = df_censored.drop(labels=['censor', 'time'], axis=1)
    y_train_c = df_censored[['censor', 'time']]

    X_train = pd.concat([X_train_c, X_train_uc])
    y_train = pd.concat([y_train_c, y_train_uc])

    uncensored_y_train = classification_transform(y_train_uc)
    uncensored_y_test = classification_transform(y_test_uc)
    datasets_dict = {'uncensored': {'X_train': X_train_uc, 'y_train': uncensored_y_train,
                                    'y_test': uncensored_y_test},
                     'censored': {'X_train': X_train, 'y_train': y_train},
                     'test': {'X': X_test_uc, 'y': y_test_uc}}
    return datasets_dict


# returns a dataframe that uses a given fraction of its censored samples (frac between 0 and 1), for a given timeframe
def use_censored_samples(X, y, frac, time: int):
    df = pd.merge(X.drop(labels=['censor', 'time'], axis=1, errors='ignore'), y, left_index=True, right_index=True, how='outer')
    df_censored = df[(df['time'] < time) & (df['censor'] == False)]
    df_uncensored = df[~((df['time'] < time) & (df['censor'] == False))]
    df = get_df_with_censoring_ratio(df_censored, df_uncensored, frac)
    num_censored = df[(df['time'] < time) & (df['censor'] == False)].shape[0]
    censored_percent = num_censored/df.shape[0]
    X = df.drop(labels=['censor', 'time'], axis=1)
    y = df[['censor', 'time']]
    return X, y


def get_df_with_censoring_ratio(df_censored, df_uncensored, r):
    # finding the frac of the censored data
    amount = r*df_uncensored.shape[0]/(1-r)
    return pd.concat([df_censored.sample(int(amount)), df_uncensored])


def load_prepare_whas500_dataset():
    X, y = load_whas500()
    orig_data = pd.merge(pd.DataFrame(X), pd.DataFrame(y), left_index=True, right_index=True, how='outer')
    orig_data.dropna(inplace=True)

    orig_data.rename(columns={'fstat': 'censor', 'lenfol': 'time'}, inplace=True)
    orig_data['age'] = orig_data['age'].astype(int)
    orig_data['diasbp'] = orig_data['diasbp'].astype(int)
    orig_data['hr'] = orig_data['hr'].astype(int)
    orig_data['los'] = orig_data['los'].astype(int)
    orig_data['time'] = orig_data['time'].astype(int)
    orig_data['sysbp'] = orig_data['sysbp'].astype(int)
    orig_data = orig_data[orig_data['time'] > 0]

    return orig_data


def load_prepare_aids_dataset(event='aids'):
    X, y = load_aids(event)
    orig_data = pd.merge(pd.DataFrame(X), pd.DataFrame(y), left_index=True, right_index=True, how='outer')
    orig_data.dropna(inplace=True)

    if event=='aids':
        orig_data.rename(columns={'fstat': 'censor', 'lenfol': 'time'}, inplace=True)
    else:
        orig_data.rename(columns={'censor_d': 'censor', 'time_d': 'time'}, inplace=True)
    orig_data['age'] = orig_data['age'].astype(int)
    orig_data['time'] = orig_data['time'].astype(int)
    orig_data['priorzdv'] = orig_data['priorzdv'].astype(int)
    orig_data = orig_data[orig_data['time'] > 0]

    return orig_data


def load_prepare_breast_cancer_rna_dataset():
    orig_data = pd.read_csv('datasets/src/breast_cancer_rna.csv')
    orig_data.dropna(inplace=True)

    orig_data.rename(columns={'Events': 'censor', 'Duration': 'time'}, inplace=True)
    orig_data['time'] = orig_data['time'].astype(float)
    orig_data = orig_data[orig_data['time'] > 0]

    return orig_data

def load_prepare_flchain_dataset():
    X, y = load_flchain()
    orig_data = pd.merge(pd.DataFrame(X), pd.DataFrame(y), left_index=True, right_index=True, how='outer')

    orig_data.rename(columns={'death': 'censor', 'futime': 'time'}, inplace=True)
    orig_data['mgus'].replace({'no': 0, 'yes': 1}, inplace=True)
    orig_data['mgus'] = orig_data['mgus'].astype('int')
    orig_data['age'] = orig_data['age'].astype('int')
    orig_data['sex'].replace({'M': 0, 'F': 1}, inplace=True)
    orig_data['sex'] = orig_data['sex'].astype('int')
    orig_data['flc.grp'] = orig_data['flc.grp'].astype('int')
    orig_data.drop(columns=['sample.yr', 'chapter'], inplace=True)
    orig_data = orig_data[orig_data['time'] > 0]
    orig_data.dropna(inplace=True)
    return orig_data