import numpy as np
import torch

import exp_utils
from data_utils import io


class _Elites:
    def __init__(self, name, exp, fitness):
        self.name = name
        self.exp = exp
        self.fitness = fitness[0]
        self.fitness_list = fitness[1]
        self.fit = fitness[1][-1]


def calculate_fitness_range():
    directory = '/home/luoyuanzhen/result/log_v2/'
    fnames = [f'kkk{i}_30log.json' for i in range(6)] + [f'feynman{i}_30log.json' for i in range(6)]

    import json
    for fname in fnames:
        file = f'{directory}{fname}'
        with open(file, 'r') as f:
            records = json.load(f)

        elite_keys = [key for key in records.keys() if key.startswith('elite')]

        fitnesses = []
        final_fits = []
        for num, key in enumerate(elite_keys):
            name = f'elite[{num}]'
            elite_dict = records[name]
            fitnesses.append(elite_dict['fitness'][0])
            final_fits.append(elite_dict['fitness'][1][-1])

        print(f'{fname}: {np.mean(fitnesses)}({np.mean(final_fits)}), {np.min(fitnesses)}({np.min(final_fits)}), {np.max(fitnesses)}({np.max(final_fits)})')


def sort_by_fitness():
    file = '/home/luoyuanzhen/result/log_v2/kkk5_30log.json'

    import json
    with open(file, 'r') as f:
        records = json.load(f)

    elite_keys = [key for key in records.keys() if key.startswith('elite')]

    elites_ = []
    for num, key in enumerate(elite_keys):
        name = f'elite[{num}]'
        elite_dict = records[name]
        elites_.append(_Elites(name, elite_dict['final_expression'], elite_dict['fitness']))

    elites_.sort(key=lambda x: x.fit)

    for elite in elites_:
        print(elite.__dict__)


def see_fitness_trend():
    file = '/home/luoyuanzhen/result/logs/'
    data_name = 'feynman4'
    fname = f'{data_name}_30cfs'
    save_name = f'/home/luoyuanzhen/result/img_v2/{data_name}_trend.pdf'
    is_log = True

    cfs = io.get_dataset(f'{file}{fname}')
    if is_log:
        cfs = torch.log2(cfs)
        ylabel = 'log2(fitness)'
    else:
        ylabel = 'fitness'
    # 30, 5000
    exp_utils.draw_f_trend(save_name, cfs.shape[0], [cfs.T.tolist()], legends=['srnn'], title=None, ylabel=ylabel)


if __name__ == '__main__':
    sort_by_fitness()
    # see_fitness_trend()
    # calculate_fitness_range()
