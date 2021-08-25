import numpy as np
import torch
from torch import nn

from CGPNet.functions import default_functions
from CGPNet.nets import CGPNet
from CGPNet.params import NetParameters
from CGPNet.utils import report, probabilistic_mutate_net


class Evolution:
    def __init__(self,
                 n_rows=5,
                 n_cols=5,
                 levels_back=None,
                 function_set=default_functions,
                 n_eph=1,
                 ):
        self.neurons = None
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.levels_back = levels_back
        if levels_back is None:
            self.levels_back = self.n_cols * self.n_rows + 1

        self.function_set = function_set
        self.n_eph = n_eph

        self.net_params = None

    @staticmethod
    def _get_protected_loss(output, y):
        loss = nn.MSELoss()(output, y)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(float('inf'))
        return loss.item()

    @staticmethod
    def _calculate_weighted_fitness(fitness_list):
        fitness = 0.0
        if len(fitness_list) == 1:
            sam_weights = [1.]
        else:
            n_hidden = len(fitness_list) - 1
            weight = 1 / n_hidden
            sam_weights = [weight for _ in range(n_hidden)] + [1.]

        for layer_fitness, weight in zip(fitness_list, sam_weights):
            fitness += weight * layer_fitness

        return fitness

    def _evaluate_fitness(self, pop, data_list, valid_data_list):
        if valid_data_list is None:
            x = data_list[0]
            nn_outputs = data_list[1:]
        else:
            x = torch.vstack((data_list[0], valid_data_list[0]))
            nn_outputs = [torch.vstack((data_list[i], valid_data_list[i])) for i in range(1, len(data_list))]
        for indiv in pop:
            predictions = indiv(x)
            indiv.fitness_list = []
            for nn_output, prediction in zip(nn_outputs, predictions):
                loss = self._get_protected_loss(nn_output, prediction)
                indiv.fitness_list.append(loss)
            indiv.fitness = self._calculate_weighted_fitness(indiv.fitness_list)

    def evolution_strategy(self, population, data_list, valid_data_list):
        # chromosome select strategy
        f_cgps, w_cgps = [], []
        num_chrom = len(population[0].f_cgps)
        chrom_input = data_list[0]

        for chrom_idx in range(num_chrom):
            chrom_losses = []
            for indiv in population:
                chrom_f, chrom_w = indiv.f_cgps[chrom_idx], indiv.w_cgps[chrom_idx]

                his = chrom_f(chrom_input) * chrom_w(chrom_input)  # n*1 n*m
                hi = data_list[chrom_idx+1]

                chrom_losses.append(self._get_protected_loss(his, hi))
            best_idx = np.argmin(chrom_losses)
            best_f, best_w = population[best_idx].f_cgps[chrom_idx], population[best_idx].w_cgps[chrom_idx]

            f_cgps.append(best_f)
            w_cgps.append(best_w)

            chrom_input = best_f(chrom_input) * best_w(chrom_input)
        # choose any individual in the population would be ok, since its layers would be replaced in the end.
        parent = population[0]
        # simpily replace its layers and fitness
        parent.f_cgps, parent.w_cgps = f_cgps, w_cgps
        # finally, evaluate its fitness using validation data
        self._evaluate_fitness([parent], data_list, valid_data_list)

        return parent

    def start(self,
              data_list,
              n_pop=200, n_gen=5000, prob=0.4, stop_fitness=1e-5,
              verbose=0, n_jobs=1,
              random_state=None,
              valid_data_list=None):
        self.neurons = [data.shape[1] for data in data_list]

        if len(data_list) != len(self.neurons):
            raise ValueError(f"Data_list's length {len(data_list)} != neurons' length {len(self.neurons)}")
        for data, n_neuron in zip(data_list, self.neurons):
            if data.shape[1] != n_neuron:
                raise ValueError(f"Shape[1] of data in data_list {data.shape[1]} != n_neuron {n_neuron}")

        self.net_params = NetParameters(
            neurons=self.neurons,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            levels_back=self.levels_back,
            function_set=self.function_set,
            n_eph=self.n_eph
        )

        conv_f, population, history_elites = [], None, []
        parent, gen = None, 0
        if verbose:
            report()
        for gen in range(1, n_gen + 1):
            if not population:
                # init
                population = [CGPNet(self.net_params) for _ in range(n_pop)]
            else:
                # mutate, note that the inital population would not be mutated
                # E(1,n-1)
                population = [parent] + \
                             [probabilistic_mutate_net(parent, prob)
                              for _ in range(n_pop - 1)]

            if parent is None:
                parent = self.evolution_strategy(population, data_list, valid_data_list)
            else:
                new_parent = self.evolution_strategy(population[1:], data_list, valid_data_list)
                parent = new_parent if new_parent.fitness < parent.fitness else parent

            conv_f.append(parent.fitness)
            _add_history_elite(history_elites, parent)

            if verbose and gen % verbose == 0:
                report(parent, gen)
            if parent.fitness <= stop_fitness:
                break

        if gen < n_gen - 1:
            condition = 'reach stop fitness'
        else:
            condition = 'reach n generation'

        if verbose:
            print(f'Stop evolution, condition:{condition}')

        # population.sort(key=lambda x: x.fitness)
        history_elites.sort(key=lambda x: x.fitness)

        return history_elites, conv_f


def _add_history_elite(history_elites, elite):
    if len(history_elites) == 0 or history_elites[-1] != elite:
        history_elites.append(elite)




