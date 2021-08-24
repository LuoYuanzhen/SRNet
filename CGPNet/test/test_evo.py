from CGPNet.evolution import Evolution
from CGPNet.utils import pretty_net_exprs
from data_utils import io


def test_evolution():
    data_list = io.get_nn_datalist('../../dataset/kkk0_nn/')

    evolution = Evolution()
    elites, conv_f = evolution.start(data_list=data_list, n_gen=500, verbose=1)

    fcgps = []
    for fcgp in elites[0].f_cgps:
        fcgps.append(fcgp.get_expressions()[0])

    wcgps = []
    for wcgp in elites[0].w_cgps:
        wcgps.append(wcgp.get_expressions())

    print('fs:', fcgps, '\n')

    print('ws', wcgps, '\n')

    print('expressions:', elites[0].get_cgp_expressions())

    print('final:', pretty_net_exprs(elites[0]))


if __name__ == '__main__':
    test_evolution()
