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
from exp_utils import save_cfs, draw_f_trend, individual_to_dict
from neural_networks.nn_models import NN_MAP

vars_map = {
    'feynman0': (['m0', 'v', 'c'], 'm'),
    'feynman1': (['q1', 'q2', 'e', 'r'], 'F'),
    'feynman2': (['G', 'm1', 'm2', 'r1', 'r2'], 'U'),
    'feynman3': (['k', 'x'], 'U'),
    'feynman4': (['G', 'c', 'm1', 'm2', 'r'], 'P'),
    'feynman5': (['q', 'y', 'V', 'd', 'e'], 'F')
}


def _train_process(controller, trainer, data_list, eps, msg, extra_data_list):
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
                                     extra_data_list=extra_data_list
                                     )
    end_time = datetime.datetime.now()

    return elites, convf, (end_time - start_time).seconds / 60


def run_a_dataset(trainer, evo_params, data_list, run_n_epoch, fname, extra_data_list=None, log_dir=None, img_dir=None, msg=''):

    clas_net, clas_cgp = clas_net_map[evo_params['clas_net']], clas_cgp_map[evo_params['clas_cgp']]
    var_names = vars_map[fname][0] if fname in vars_map else None
    controller = Evolution(n_rows=evo_params['n_rows'],
                           n_cols=evo_params['n_cols'],
                           levels_back=evo_params['levels_back'],
                           function_set=evo_params['function_set'],
                           n_eph=evo_params['n_eph'],
                           clas_net=clas_net,
                           clas_cgp=clas_cgp)

    results = Parallel(n_jobs=run_n_epoch)(
        delayed(_train_process)(controller,
                                trainer,
                                data_list,
                                evo_params,
                                f'{fname}-{msg}-{epoch} start:\n',
                                extra_data_list
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


def generate_extrapolation_data(num_sample, inter_domains, extra_domains):
    extra_input = []
    for inter, extra in zip(inter_domains, extra_domains):
        extra1, extra2 = (extra[0], inter[0]), (inter[1], extra[1])
        extra_data1 = (extra1[1] - extra1[0]) * torch.rand(num_sample // 2, 1) + extra1[0]
        extra_data2 = (extra2[1] - extra2[0]) * torch.rand(num_sample // 2, 1) + extra2[1]
        extra_input.append(torch.vstack((extra_data1, extra_data2)))
    extra_input = torch.hstack(extra_input)

    return extra_input


def run_all_experiments(trainer, evo_params, all_names, data_dir, log_dir, img_dir, xlabel=None, run_n_epoch=30):
    srnn_fs_list = []
    for fname in all_names:
        var_names = vars_map[fname][0] if fname in vars_map else None

        # run dcgp net without save
        nn_dir = f'{data_dir}{fname}_nn/'
        nn_data_list = io.get_nn_datalist(nn_dir)

        extra_data_list = None
        if evo_params['extra_select']:
            # generate extrapolation data for model selection
            nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[fname]).cpu()

            inter_domains, extra_domains = INTER_MAP[fname], VALID_MAP[fname]
            num_sample = max(nn_data_list[0].shape[0] // 10, 10)
            extra_input = generate_extrapolation_data(num_sample, inter_domains, extra_domains)

            extra_data_list = [extra_input] + list(nn(extra_input))

        srnn_cfs, srnn_fs, srnn_ts, srnn_elites = run_a_dataset(trainer,
                                                                evo_params,
                                                                nn_data_list,
                                                                run_n_epoch,
                                                                fname,
                                                                extra_data_list=extra_data_list)

        # run dcgp net w.r.t nn's input and output without save
        # true_file = f'{data_dir}{fname}'
        # true_data = io.get_dataset(true_file)
        # true_data_list = [true_data[:, :-1], true_data[:, -1:]]
        # sr_cfs, sr_fs, sr_ts, sr_elites = run_a_dataset(trainer, evo_params, true_data_list, run_n_epoch, fname,
        #                                                 msg='sr')

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
            # 'sr_mean_time': np.mean(sr_ts),
            # 'sr_mean_fitness': np.mean(sr_fs),
            # 'sr_min_fitness': np.min(sr_fs),
            # 'sr_max_fitness': np.max(sr_fs),
            # 'sr_fitness': sr_fs,
            # 'sr_elite': individual_to_dict(sr_elites[0], var_names)
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
    if len(sys.argv) < 2:
        # we can test experiments by directly run this program
        data_dir, xlabel = 'dataset/', 'K'
        log_dir, img_dir = '/home/luoyuanzhen/result/log_extra/', '/home/luoyuanzhen/result/img_extra/'
    else:
        data_dir, xlabel = sys.argv[1], sys.argv[2]
        log_dir, img_dir = sys.argv[3], sys.argv[4]

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
        'n_population': 200,
        'n_generation': 5000,
        'prob': 0.4,
        'verbose': 10,
        'stop_fitness': 1e-5,
        'random_state': None,
        'n_jobs': 1,
        'n_epoch': 0,
        'end_to_end': False,
        'extra_select': True,
        'evolution_strategy': 'fitness_select'
    }
    trainer = clas_optim_map[evo_params['optim']](end_to_end=evo_params['end_to_end'])

    all_names = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
                 'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']
    run_all_experiments(trainer, evo_params, all_names, data_dir, log_dir, img_dir, xlabel)

    # fname = 'kkk0'
    # data_list = io.get_nn_datalist(f'{data_dir}{fname}_nn/')
    # run_a_dataset(trainer, evo_params, data_list, 1, fname, log_dir=log_dir, img_dir=img_dir)





