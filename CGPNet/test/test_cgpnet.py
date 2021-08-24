import torch
import sympy as sp

from CGPNet.cgp import WeightCGP
from CGPNet.functions import default_functions
from CGPNet.nets import CGPNet
from CGPNet.params import CGPParameters, NetParameters
from CGPNet.utils import pretty_net_exprs, probabilistic_mutate_net
from data_utils import io


def test_WeightCGP():
    x = torch.normal(mean=0., std=1., size=(5, 3))
    wcgp_params = CGPParameters(
        n_inputs=x.shape[1]+1,
        n_outputs=3,
        n_rows=5,
        n_cols=5,
        levels_back=None,
        function_set=default_functions,
        n_eph=1
    )

    weightCGP = WeightCGP(wcgp_params)

    print(x)
    print(weightCGP(x))
    print(weightCGP.get_expressions())


def test_CGPNet():
    # test intialization, expressions, pretty_print, call
    data_list = io.get_nn_datalist('../../dataset/kkk0_nn/')
    neurons = [data.shape[1] for data in data_list]

    net_params = NetParameters(
        neurons=neurons,
        n_rows=5,
        n_cols=5,
        levels_back=None,
        function_set=default_functions,
        n_eph=1
    )

    cgp_net = CGPNet(net_params)

    def _report_net(_net):
        fcgps = []
        for fcgp in _net.f_cgps:
            fcgps.append(fcgp.get_expressions()[0])

        wcgps = []
        for wcgp in _net.w_cgps:
            wcgps.append(wcgp.get_expressions())

        print('fs:', fcgps, '\n')
        print('ws', wcgps, '\n')

        print('expressions:', _net.get_cgp_expressions(), '\n')

        print(pretty_net_exprs(_net))

        print('net(x):', _net(data_list[0]))

    print('original')
    _report_net(cgp_net)

    # mutate
    parent = probabilistic_mutate_net(cgp_net, 0.4)
    print('mutated cgpnet')
    _report_net(parent)


if __name__ == '__main__':
    # test_WeightCGP()
    test_CGPNet()
