import torch

from CGPNet.cgp import CGP, WeightCGP, create_cgp_by_net_params
from CGPNet.params import NetParameters
from CGPNet.utils import layer_expression


class CGPNet:
    def __init__(self, net_params: NetParameters, f_cgps=None, w_cgps=None):
        self.net_params = net_params

        self.neurons = net_params.neurons
        self.n_layer = len(self.neurons)

        self.fitness = None
        self.fitness_list = []

        self.f_cgps = []
        self.w_cgps = []
        if f_cgps and w_cgps:
            self.f_cgps = f_cgps
            self.w_cgps = w_cgps
        else:
            for i in range(1, self.n_layer):
                self.f_cgps.append(create_cgp_by_net_params(self.neurons[i-1], 1, net_params, CGP))
                self.w_cgps.append(create_cgp_by_net_params(self.neurons[i-1]+1, self.neurons[i], net_params, WeightCGP))

    def __call__(self, x):
        his_input = x
        his = []
        for f_cgp, w_cgp in zip(self.f_cgps, self.w_cgps):
            his_input = f_cgp(his_input) * w_cgp(his_input)
            his.append(his_input)

        return his

    def get_cgp_expressions(self):
        fs, ws = [], []
        for fcgp in self.f_cgps:
            fs.append(fcgp.get_expressions()[0])  # it only has one expression

        for wcgp in self.w_cgps:
            ws.append(wcgp.get_expressions())  # w(x, i) has self.n_outputs expressions

        exps = []
        # each f and w need to multiply together
        for f, w in zip(fs, ws):
            exps.append(layer_expression(f, w))

        return exps

    def generate_offspring(self, f_gidxs_list, w_gidxs_list, f_mutant_genes_list, w_mutant_genes_list):
        f_cgps, w_cgps = [], []
        for f, w, f_gidxs, f_mutant_genes, w_idxs, w_mutant_genes in zip(
                self.f_cgps, self.w_cgps, f_gidxs_list, f_mutant_genes_list, w_gidxs_list, w_mutant_genes_list):
            f_cgps.append(f.generate_offspring(f_gidxs, f_mutant_genes))
            w_cgps.append(w.generate_offspring(w_idxs, w_mutant_genes))
        return CGPNet(self.net_params, f_cgps, w_cgps)

    @classmethod
    def encode_net(cls,
                   net_params: NetParameters,
                   f_genes_list, f_ephs_list, w_genes_list, w_ephs_list):
        """class method that encode the net based on existing params"""
        neurons = net_params.neurons
        n_layers = len(neurons)

        if len(f_genes_list) != len(f_ephs_list) != len(w_genes_list) != len(w_ephs_list) and n_layers != len(f_genes_list) + 1:
            raise ValueError('length of genes, ephs, ws should all be equal to n_layer - 1!')

        f_cgps, w_cgps = [], []
        for i in range(1, n_layers):
            fgenes, fephs = f_genes_list[i - 1], torch.tensor(f_ephs_list[i - 1])
            wgens, wephs = torch.tensor(w_genes_list[i - 1]), torch.tensor(w_ephs_list[i - 1])

            f_cgps.append(create_cgp_by_net_params(neurons[i-1], 1, net_params, CGP, fgenes, fephs))
            w_cgps.append(create_cgp_by_net_params(neurons[i-1]+1, neurons[i], net_params, WeightCGP, wgens, wephs))

        return cls(net_params, f_cgps, w_cgps)
