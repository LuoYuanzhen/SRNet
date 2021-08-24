import datetime
import json

import joblib
import numpy as np
from joblib import Parallel, delayed

from CGPNet.evolution import Evolution
from CGPNet.functions import default_functions
from CGPNet.utils import pretty_net_exprs
from data_utils import io, draw
from dataset_config import VALID_MAP
from exp_utils import save_cfs, draw_f_trend, generate_domains_data
from neural_networks.nn_models import NN_MAP

data_dir, xlabel = 'dataset/', 'F'
log_dir, img_dir = 'cgpnet_result/wnx_logs/', 'cgpnet_result/wnx_imgs/'

io.mkdir(log_dir)
io.mkdir(img_dir)

evo_params = {
    'n_rows': 5,
    'n_cols': 5,
    'levels_back': None,
    'function_set': default_functions,
    'n_eph': 1,

    'n_population': 200,
    'n_generation': 100,
    'prob': 0.4,
    'verbose': 10,
    'stop_fitness': 1e-5,
    'random_state': None,
    'n_jobs': 1,
    'validation': True
}

all_names = ['kkk0', 'kkk1']
run_epoch = 3


vars_map = {
    'feynman0': (['m0', 'v', 'c'], 'm'),
    'feynman1': (['q1', 'q2', 'e', 'r'], 'F'),
    'feynman2': (['G', 'm1', 'm2', 'r1', 'r2'], 'U'),
    'feynman3': (['k', 'x'], 'U'),
    'feynman4': (['G', 'c', 'm1', 'm2', 'r'], 'P'),
    'feynman5': (['q', 'y', 'V', 'd', 'e'], 'F')
}


def individual_to_dict(indiv, var_names=None):
    end_exp = pretty_net_exprs(indiv, var_names)
    f_cgp_genes, f_cgp_ephs = [], []
    w_cgp_genes, w_cgp_ephs = [], []
    fs, ws = [], []
    for f_cgp, w_cgp in zip(indiv.f_cgps, indiv.w_cgps):
        f_cgp_genes.append(f_cgp.get_genes())
        f_cgp_ephs.append(f_cgp.get_ephs().numpy().tolist())

        w_cgp_genes.append(w_cgp.get_genes())
        w_cgp_ephs.append(w_cgp.get_ephs().numpy().tolist())

        fs.append(f_cgp.get_expressions()[0])
        ws.append(w_cgp.get_expressions())

    indiv_dict = {'final_expression': str(end_exp),
                  'fitness': (indiv.fitness, indiv.fitness_list),
                  'expressions': str(indiv.get_cgp_expressions()),
                  'fs': str(fs),
                  'ws': str(ws),
                  'f_constants': f_cgp_ephs,
                  'f_genes': f_cgp_genes,
                  'w_constants': w_cgp_ephs,
                  'w_genes': w_cgp_genes
                  }
    return indiv_dict


def _train_process(controller, data_list, eps, msg, valid_data_list):
    print(msg)
    start_time = datetime.datetime.now()
    elites, convf = controller.start(data_list=data_list,
                                     n_pop=eps['n_population'],
                                     n_gen=eps['n_generation'],
                                     prob=eps['prob'],
                                     verbose=eps['verbose'],
                                     n_jobs=eps['n_jobs'],
                                     stop_fitness=eps['stop_fitness'],
                                     random_state=eps['random_state'],
                                     valid_data_list=valid_data_list
                                     )
    end_time = datetime.datetime.now()

    return elites, convf, (end_time - start_time).seconds / 60


def run_all_experiments():
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

        controller = Evolution(n_rows=evo_params['n_rows'],
                               n_cols=evo_params['n_cols'],
                               levels_back=evo_params['levels_back'],
                               function_set=evo_params['function_set'],
                               n_eph=evo_params['n_eph'])

        results = Parallel(n_jobs=run_epoch)(
            delayed(_train_process)(controller,
                                    nn_data_list,
                                    evo_params,
                                    f'{fname}-{epoch} start:\n',
                                    valid_data_list
                                    )
            for epoch in range(run_epoch))

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

        srnn_fs_list.append(fs)

        log_dict = {
            'name': fname,
            'evolution_parameters': evo_params,
            'neurons': list([data.shape[1] for data in nn_data_list]),
            'srnn_mean_time': np.mean(ts),
            'srnn_mean_fitness': np.mean(fs),
            'srnn_min_fitness': np.min(fs),
            'srnn_max_fitness': np.max(fs),
            'srnn_fitness': fs,
        }

        elite_results = Parallel(n_jobs=joblib.cpu_count())(
            delayed(individual_to_dict)(elite, var_names)
            for elite in elites)

        for num, result in enumerate(elite_results):
            log_dict[f'elite[{num}]'] = result

        with open(f'{log_dir}{fname}_30log.json', 'w') as f:
            json.dump(log_dict, f, indent=4)

        save_cfs(f'{log_dir}{fname}_30cfs', cfs)
        draw_f_trend(f'{img_dir}{fname}_trend.pdf', evo_params['n_generation'], [cfs], legends=['srnn'], title=fname)

    draw.draw_fitness_box(f'{img_dir}{xlabel}_box_fit.pdf', srnn_fs_list, xlabel=xlabel)


run_all_experiments()

