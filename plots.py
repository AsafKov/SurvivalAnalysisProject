import matplotlib.pyplot as plt
import data_processing
import numpy as np


# plot a line plot of aucs by censored percentage used in training
def line_plot(mean_rsf_aucs, mean_rfc_auc, censored_percentage, num_experiments, title):
    plt.plot(censored_percentage, mean_rsf_aucs, label='rfs auc')
    plt.scatter(0, mean_rfc_auc, color='red', label='rfc auc')
    plt.legend()
    plt.xlabel('Censored Percentage')
    plt.ylabel('AUC')
    plt.title(f"mean AUC scores across {num_experiments} experiments - {title}")
    plt.grid(True)
    plt.rcParams["figure.facecolor"] = "w"
    plt.show()


# plot a line plot of censored percentage by time of dataframe
def censored_by_time_plot(df):
    X = []
    y = []
    for t in np.arange(df['time'].min(), df['time'].max(), int((df['time'].max()-df['time'].min()))/100):
        X.append(t)
        y.append(data_processing.censored_percentage(df, t))
    plt.plot(X, y)
    plt.xlabel('time')
    plt.ylabel('percentage')
    plt.title(f"censored samples percentage by time")
    plt.grid(True)
    plt.show()

# plot a line plot of censored percentage by time of dataframe
def censored_samples_by_time(df, time, dataset):
    X = []
    y = []
    samples = 0
    for t in np.sort(df['time'].unique()):
        X.append(t)
        samples += df[(df['time'] < time) & (df['censor'] == False) & (df['time'] == t)].shape[0]
        y.append(samples)
    fig = plt.figure(figsize=(20, 6))
    plt.plot(X, y)
    plt.xlabel('time')
    plt.ylabel('samples')
    plt.title(f"{dataset} dataset samples by time (censore time={time})")
    plt.grid(True)
    plt.show()