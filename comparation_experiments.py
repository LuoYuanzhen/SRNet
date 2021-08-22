import json
from functools import partial

import numpy as np
import pandas as pd
import scipy
import torch
from lime import lime_tabular

import exp_utils
from data_utils import io
from exp_utils import load_test_valid_data, encode_individual_from_json
from maple.MAPLE import MAPLE
from maple.Misc import normaliza_data, unpack_coefs
from neural_networks.nn_models import NN_MAP


datasets = ['kkk1']
trials = [1]


def my_pred(model, x):

    return model(torch.from_numpy(x).float())[-1].detach().numpy().ravel()


def run_compare(dataset, trial):
    # Hyperparamaters
    num_perturbations = 5

    # Fixes an issue where threads of inherit the same rng state
    scipy.random.seed()

    # Outpt
    out = {}
    file = open("ComparationTrials/" + dataset + "_" + str(trial) + ".json", "w")

    # Load data
    train = io.get_dataset('dataset/' + dataset)
    test, valid = load_test_valid_data(dataset)

    train, train_mean, train_stddev = normaliza_data(train)
    test, test_mean, test_stddev = normaliza_data(test)
    valid, valid_mean, valid_stddev = normaliza_data(valid)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    X_valid, y_valid = valid[:, :-1], valid[:, -1]

    n = X_train.shape[0]
    d = X_train.shape[1]

    nn_model = io.load_nn_model('dataset/'+dataset+'_nn/nn_module.pt', load_type='dict', nn=NN_MAP[dataset]).cpu()
    srnet = encode_individual_from_json(f'/home/luoyuanzhen/result/log_extra/{dataset}_30log.json', 'elite[0]')
    out["model_test_rmse"] = np.sqrt(np.mean((y_test - my_pred(nn_model, X_test))**2)).astype(float)
    out["srnet_test_rmse"] = np.sqrt(np.mean((my_pred(nn_model, X_test) - my_pred(srnet, X_test))**2)).astype(float)

    # Fit LIME and MAPLE explainers to the model
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=False, mode="regression")
    exp_maple = MAPLE(X_train, my_pred(nn_model, X_train), X_valid, my_pred(nn_model, X_valid))

    lime_rmse = np.zeros(1)
    maple_rmse = np.zeros(1)
    srnet_rmse = np.zeros(1)

    # save predictions of all interpretable model
    x_noise = []
    model_output, srnet_output, lime_output, maple_output = [], [], [], []
    for i in range(n):
        x = X_train[i, :]

        coefs_lime = unpack_coefs(exp_lime, x, partial(my_pred, nn_model), d, X_train)  # Allow full number of features

        e_maple = exp_maple.explain(x)
        coefs_maple = e_maple["coefs"]

        model_pred = my_pred(nn_model, x.reshape(1, -1))
        srnet_pred = my_pred(srnet, x.reshape(1, -1))
        lime_pred = np.dot(np.insert(x, 0, 1), coefs_lime)
        maple_pred = np.dot(np.insert(x, 0, 1), coefs_maple)

        model_output.append(model_pred.reshape(1, -1))
        srnet_output.append(srnet_pred.reshape(1, -1))
        lime_output.append(lime_pred.reshape(1, -1))
        maple_output.append(maple_pred.reshape(1, -1))

        lime_rmse += (lime_pred - model_pred) ** 2
        maple_rmse += (maple_pred - model_pred) ** 2
        srnet_rmse += (srnet_pred - model_pred) ** 2

    lime_rmse /= n
    maple_rmse /= n
    srnet_rmse /= n

    lime_rmse = np.sqrt(lime_rmse)
    maple_rmse = np.sqrt(maple_rmse)
    srnet_rmse = np.sqrt(srnet_rmse)

    out["lime_rmse"] = lime_rmse[0]
    out["maple_rmse"] = maple_rmse[0]
    out["srnet_rmse"] = srnet_rmse[0]

    model_output = np.vstack(model_output)
    srnet_output = np.vstack(srnet_output)
    lime_output = np.vstack(lime_output)
    maple_output = np.vstack(maple_output)

    json.dump(out, file, indent=4)
    file.close()

    ys = [model_output, srnet_output, lime_output, maple_output]
    labels = ['MLP', 'CGPNet', 'LIME', 'MAPLE']
    exp_utils.draw_project_output_scatter(X_train, ys, labels)
    exp_utils.draw_output_compare_curves(X_train[:, 0], ys, labels, n_var=X_train.shape[1])


###
# Merge Results
###
for dataset, trial in zip(datasets, trials):
    run_compare(dataset, trial)

with open("ComparationTrials/" + datasets[0] + "_" + str(trials[0]) + ".json") as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(0, index=datasets, columns=columns)

for dataset in datasets:
    for trial in trials:
        with open("ComparationTrials/" + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df.loc[dataset, name] += data[name] / len(trials)

df.to_csv("results.csv")
