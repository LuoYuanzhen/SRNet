import json
from functools import partial

import numpy as np
import pandas as pd
import scipy
import torch
from lime import lime_tabular

from data_utils import io
from exp_utils import load_test_valid_data, encode_individual_from_json_v2
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

    scales = [0.1, 0.25]
    scales_len = len(scales)

    nn_model = io.load_nn_model('dataset/'+dataset+'_nn/nn_module.pt', load_type='dict', nn=NN_MAP[dataset]).cpu()
    srnet = encode_individual_from_json_v2(f'/home/luoyuanzhen/result/log_v2/{dataset}_30log.json', 'elite[0]')
    out["model_rmse"] = np.sqrt(np.mean((y_test - my_pred(nn_model, X_test))**2)).astype(float)
    out["srnet_rmse"] = np.sqrt(np.mean((y_test - my_pred(srnet, X_test))**2)).astype(float)

    # Fit LIME and MAPLE explainers to the model
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=False, mode="regression")
    exp_maple = MAPLE(X_train, my_pred(nn_model, X_train), X_valid, my_pred(nn_model, X_valid))

    # Evaluate model faithfullness on the test set
    lime_rmse = np.zeros((scales_len))
    maple_rmse = np.zeros((scales_len))
    srnet_rmse = np.zeros((scales_len))

    # save predictions of all interpretable model
    x_noise = []
    model_output, srnet_output, lime_output, maple_output = [], [], [], []
    for i in range(n):
        x = X_test[i, :]

        coefs_lime = unpack_coefs(exp_lime, x, partial(my_pred, nn_model), d, X_train)  # Allow full number of features

        e_maple = exp_maple.explain(x)
        coefs_maple = e_maple["coefs"]

        for j in range(num_perturbations):

            noise = np.random.normal(loc=0.0, scale=1.0, size=d)

            for k in range(scales_len):
                scale = scales[k]

                x_pert = x + scale * noise

                model_pred = my_pred(nn_model, x_pert.reshape(1, -1))
                srnet_pred = my_pred(srnet, x_pert.reshape(1, -1))
                lime_pred = np.dot(np.insert(x_pert, 0, 1), coefs_lime)
                maple_pred = np.dot(np.insert(x_pert, 0, 1), coefs_maple)

                lime_rmse[k] += (lime_pred - model_pred) ** 2
                maple_rmse[k] += (maple_pred - model_pred) ** 2
                srnet_rmse[k] += (srnet_pred - model_pred) ** 2

    lime_rmse /= n * num_perturbations
    maple_rmse /= n * num_perturbations
    srnet_rmse /= n * num_perturbations

    lime_rmse = np.sqrt(lime_rmse)
    maple_rmse = np.sqrt(maple_rmse)
    srnet_rmse = np.sqrt(srnet_rmse)

    out["lime_rmse_0.1"] = lime_rmse[0]
    out["maple_rmse_0.1"] = maple_rmse[0]
    out["srnet_rmse_0.1"] = srnet_rmse[0]

    out["lime_rmse_0.25"] = lime_rmse[1]
    out["maple_rmse_0.25"] = maple_rmse[1]
    out["srnet_rmse_0.25"] = srnet_rmse[1]

    json.dump(out, file, indent=4)
    file.close()

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
