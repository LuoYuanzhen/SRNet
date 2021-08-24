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


datasets = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5']
trials = [1, 2, 3, 4, 5, 6]


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

    n = X_test.shape[0]
    d = X_train.shape[1]

    nn_model = io.load_nn_model('dataset/'+dataset+'_nn/nn_module.pt', load_type='dict', nn=NN_MAP[dataset]).cpu()
    srnet = encode_individual_from_json(f'cgpnet_result/logs/{dataset}_30log.json', 'elite[0]')
    out["model_test_rmse"] = np.sqrt(np.mean((y_test - my_pred(nn_model, X_test))**2)).astype(float)
    out["srnet_valid_rmse"] = np.sqrt(np.mean((my_pred(nn_model, X_valid) - my_pred(srnet, X_valid))**2)).astype(float)

    # Fit LIME and MAPLE explainers to the model
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=False, mode="regression")
    exp_maple = MAPLE(X_train, my_pred(nn_model, X_train), X_valid, my_pred(nn_model, X_valid))

    lime_rmse = np.zeros(2)
    maple_rmse = np.zeros(2)
    srnet_rmse = np.zeros(2)

    # save predictions of all interpretable model
    model_test_output, srnet_test_output, lime_test_output, maple_test_output = [], [], [], []
    model_valid_output, lime_valid_output, maple_valid_output = [], [], []
    for i in range(n):
        x_test = X_test[i, :]
        x_valid = X_valid[i, :]

        coefs_lime = unpack_coefs(exp_lime, x_test, partial(my_pred, nn_model), d, X_train)  # Allow full number of features

        e_maple = exp_maple.explain(x_test)
        coefs_maple = e_maple["coefs"]

        model_valid_pred = my_pred(nn_model, x_valid.reshape(1, -1))
        lime_valid_pred = np.dot(np.insert(x_valid, 0, 1), coefs_lime)
        maple_valid_pred = np.dot(np.insert(x_valid, 0, 1), coefs_maple)

        model_test_pred = my_pred(nn_model, x_test.reshape(1, -1))
        srnet_test_pred = my_pred(srnet, x_test.reshape(1, -1))
        lime_test_pred = np.dot(np.insert(x_test, 0, 1), coefs_lime)
        maple_test_pred = np.dot(np.insert(x_test, 0, 1), coefs_maple)

        model_valid_output.append(model_valid_pred.reshape(1, -1))
        lime_valid_output.append(lime_valid_pred.reshape(1, -1))
        maple_valid_output.append(maple_valid_pred.reshape(1, -1))

        model_test_output.append(model_test_pred.reshape(1, -1))
        srnet_test_output.append(srnet_test_pred.reshape(1, -1))
        lime_test_output.append(lime_test_pred.reshape(1, -1))
        maple_test_output.append(maple_test_pred.reshape(1, -1))

        lime_rmse[1] += (lime_valid_pred - model_valid_pred) ** 2
        maple_rmse[1] += (maple_valid_pred - model_valid_pred) ** 2

        lime_rmse[0] += (lime_test_pred - model_test_pred) ** 2
        maple_rmse[0] += (maple_test_pred - model_test_pred) ** 2
        srnet_rmse[0] += (srnet_test_pred - model_test_pred) ** 2

    lime_rmse /= n
    maple_rmse /= n
    srnet_rmse /= n

    lime_rmse = np.sqrt(lime_rmse)
    maple_rmse = np.sqrt(maple_rmse)
    srnet_rmse = np.sqrt(srnet_rmse)

    out["lime_valid_rmse"] = lime_rmse[1]
    out["maple_valid_rmse"] = maple_rmse[1]

    out["srnet_test_rmse"] = srnet_rmse[0]
    out["lime_test_rmse"] = lime_rmse[0]
    out["maple_test_rmse"] = maple_rmse[0]

    model_test_output = np.vstack(model_test_output)
    srnet_test_output = np.vstack(srnet_test_output)
    lime_test_output = np.vstack(lime_test_output)
    maple_test_output = np.vstack(maple_test_output)

    json.dump(out, file, indent=4)
    file.close()

    ys = [model_test_output, srnet_test_output, lime_test_output, maple_test_output]
    labels = ['MLP', 'CGPNet', 'LIME', 'MAPLE']
    exp_utils.draw_project_output_scatter(X_test, ys, labels)

    x_sorted, indexs = torch.sort(torch.from_numpy(X_test[:, 0]))
    _, reback_indexs = torch.sort(indexs)
    for y in ys:
        y = torch.from_numpy(y).index_select(0, reback_indexs)

    exp_utils.draw_output_compare_curves(X_test[:, 0], ys, labels, n_var=X_train.shape[1])


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
