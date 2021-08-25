"""Experiment codes for CGPNet training."""
import datetime
import json
import sys

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from dCGPNet.config import clas_net_map, clas_cgp_map, clas_optim_map
from dCGPNet.functions import default_functions
from dCGPNet.methods import Evolution
from data_utils import io, draw
from dataset_config import INTER_MAP, VALID_MAP
from exp_utils import save_cfs, draw_f_trend, individual_to_dict, generate_domains_data
from neural_networks.nn_models import NN_MAP

vars_map = {
    'feynman0': (['m0', 'v', 'c'], 'm'),
    'feynman1': (['q1', 'q2', 'e', 'r'], 'F'),
    'feynman2': (['G', 'm1', 'm2', 'r1', 'r2'], 'U'),
    'feynman3': (['k', 'x'], 'U'),
    'feynman4': (['G', 'c', 'm1', 'm2', 'r'], 'P'),
    'feynman5': (['q', 'y', 'V', 'd', 'e'], 'F')
}


def _train_process(controller, trainer, data_list, eps, msg, valid_data_list):
    print(msg)
    start_time = datetime.datetime.now()
    elites, convf = controller.start(data_list=data_list,
                                     trainer=trainer,
                                     n_pop=eps['n_population'],
                                     n_gen=eps['n_generation'],
                                     prob=eps['prob'],
                                     verbose=eps['verbose'],
                                     n_jobs=eps['n_jobs'],
                                     stop_fitness=eps['stop_fitness'],
                                     random_state=eps['random_state'],
                                     evo_strategy=eps['evolution_strategy'],
                                     valid_data_list=valid_data_list
                                     )
    end_time = datetime.datetime.now()

    return elites, convf, (end_time - start_time).seconds / 60


def run_a_dataset(trainer, evo_params, data_list, run_n_epoch, fname, valid_data_list=None, log_dir=None, img_dir=None, msg=''):
    clas_net, clas_cgp = clas_net_map[evo_params['clas_net']], clas_cgp_map[evo_params['clas_cgp']]
    var_names = vars_map[fname][0] if fname in vars_map else None
    controller = Evolution(n_rows=evo_params['n_rows'],
                           n_cols=evo_params['n_cols'],
                           levels_back=evo_params['levels_back'],
                           function_set=evo_params['function_set'],
                           n_eph=evo_params['n_eph'],
                           clas_net=clas_net,
                           clas_cgp=clas_cgp,
                           add_bias=evo_params['add_bias'])

    results = Parallel(n_jobs=run_n_epoch)(
        delayed(_train_process)(controller,
                                trainer,
                                data_list,
                                evo_params,
                                f'{fname}-{msg}-{epoch} start:\n',
                                valid_data_list
                                )
        for epoch in range(run_n_epoch))

    fs, ts = [], []  # for log
    cfs = []  # for trend draw
    elites = []  # All top10 best elites from each runtimes. For log
    for result in results:
        process_elites, convf, time = result
        fs.append(process_elites[0].fitness)
        ts.append(time)
        cfs.append(convf)
        elites += process_elites[:min(10, len(process_elites))]

    elites.sort(key=lambda x: x.fitness)

    if log_dir:
        log_dict = {'name': fname,
                    'evolution_parameters': evo_params,
                    'neurons': list([data.shape[1] for data in data_list]),
                    'mean_time': np.mean(ts),
                    'mean_fitness': np.mean(fs),
                    'min_fitness': np.min(fs),
                    'max_fitness': np.max(fs),
                    'fitness': fs
                    }

        elite_results = Parallel(n_jobs=joblib.cpu_count())(
            delayed(individual_to_dict)(elite, var_names)
            for elite in elites)

        for num, result in enumerate(elite_results):
            log_dict[f'elite[{num}]'] = result

        with open(f'{log_dir}{fname}_30log.json', 'w') as f:
            json.dump(log_dict, f, indent=4)

        save_cfs(f'{log_dir}{fname}_30cfs', cfs)

    if img_dir:
        draw_f_trend(f'{img_dir}{fname}_trend.pdf', evo_params['n_generation'], [cfs], legends=['srnn'], title=fname)

    return cfs, fs, ts, elites


def run_all_experiments(trainer, evo_params, all_names, data_dir, log_dir, img_dir, xlabel=None, run_n_epoch=30):
    srnn_fs_list = []
    for fname in all_names:
        var_names = vars_map[fname][0] if fname in vars_map else None

        # run dcgp net without save
        nn_dir = f'{data_dir}{fname}_nn/'
        nn_data_list = io.get_nn_datalist(nn_dir)

        valid_data_list = None
        if evo_params['validation']:
            # generate extrapolation data for model selection
            nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[fname]).cpu()

            num_sample = max(nn_data_list[0].shape[0] // 10 * 3, 10)  # train:valid = 7:3
            valid_input = generate_domains_data(num_sample, VALID_MAP[fname])

            valid_data_list = [valid_input] + list(nn(valid_input))
        srnn_cfs, srnn_fs, srnn_ts, srnn_elites = run_a_dataset(trainer,
                                                                evo_params,
                                                                nn_data_list,
                                                                run_n_epoch,
                                                                fname,
                                                                valid_data_list=valid_data_list)

        srnn_fs_list.append(srnn_fs)

        log_dict = {
            'name': fname,
            'evolution_parameters': evo_params,
            'neurons': list([data.shape[1] for data in nn_data_list]),
            'srnn_mean_time': np.mean(srnn_ts),
            'srnn_mean_fitness': np.mean(srnn_fs),
            'srnn_min_fitness': np.min(srnn_fs),
            'srnn_max_fitness': np.max(srnn_fs),
            'srnn_fitness': srnn_fs,
        }

        elite_results = Parallel(n_jobs=joblib.cpu_count())(
            delayed(individual_to_dict)(elite, var_names)
            for elite in srnn_elites)

        for num, result in enumerate(elite_results):
            log_dict[f'elite[{num}]'] = result

        with open(f'{log_dir}{fname}_30log.json', 'w') as f:
            json.dump(log_dict, f, indent=4)

        save_cfs(f'{log_dir}{fname}_30cfs', srnn_cfs)
        draw_f_trend(f'{img_dir}{fname}_trend.pdf', evo_params['n_generation'], [srnn_cfs], legends=['srnn'], title=fname)

    draw.draw_fitness_box(f'{img_dir}{xlabel}_box_fit.pdf', srnn_fs_list, xlabel=xlabel)


if __name__ == '__main__':

    data_dir, xlabel = 'dataset/', 'K'
    log_dir, img_dir = 'cgpnet_result/logs/', 'cgpnet_result/imgs/'

    io.mkdir(log_dir)
    io.mkdir(img_dir)

    evo_params = {
        'clas_net': 'OneVectorCGPNet',
        'clas_cgp': 'OneExpOneOutCGPLayer',
        'optim': 'Newton',
        'n_rows': 5,
        'n_cols': 5,
        'levels_back': None,
        'function_set': default_functions,
        'n_eph': 1,
        'add_bias': False,

        'n_population': 200,
        'n_generation': 5000,
        'prob': 0.4,
        'verbose': 10,
        'stop_fitness': 1e-5,
        'random_state': None,
        'n_jobs': 1,
        'n_epoch': 0,
        'end_to_end': False,
        'validation': True,
        'evolution_strategy': 'chromosome_select'
    }
    trainer = clas_optim_map[evo_params['optim']](end_to_end=evo_params['end_to_end'])

    all_names = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
                 'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']
    run_all_experiments(trainer, evo_params, all_names, data_dir, log_dir, img_dir, xlabel)

    # fname = 'kkk0'
    # data_list = io.get_nn_datalist(f'{data_dir}{fname}_nn/')
    #
    # nn_dir = f'{data_dir}{fname}_nn/'
    # nn_data_list = io.get_nn_datalist(nn_dir)
    #
    # valid_data_list = None
    # if evo_params['validation']:
    #     # generate extrapolation data for model selection
    #     nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[fname]).cpu()
    #
    #     num_sample = max(nn_data_list[0].shape[0] // 10 * 3, 10)  # train:valid = 7:3
    #     valid_input = generate_validation_data(num_sample, VALID_MAP[fname])
    #
    #     valid_data_list = [valid_input] + list(nn(valid_input))
    #
    # run_a_dataset(trainer, evo_params, data_list, 1, fname, valid_data_list=valid_data_list, log_dir=log_dir, img_dir=img_dir)





