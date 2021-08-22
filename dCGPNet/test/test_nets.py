import sympy as sp
import torch

from dCGPNet.functions import default_functions
from dCGPNet.nets import MulExpCGPLayer, OneLinearCGPNet, OneExpCGPLayer, OneVectorCGPNet
from dCGPNet.params import CGPParameters, NetParameters
from dCGPNet.utils import pretty_net_exprs


def _print_net(net, x):
    print('###expressions:')
    print(net.get_expressions())

    results = net(x)
    print('###results:')
    [print(result, result.shape) for result in results]

    w_list, bias_list = net.get_ws(), net.get_bias()
    print('###weights:')
    [print(w, w.shape) for w in w_list]

    print('###biases:')
    [print(bias, bias.shape) for bias in bias_list]

    print('###parameters:')
    print(net.get_net_parameters())

    print('###final expression:')
    # print(get_net_expression(w_list, net.get_expressions()))


def test_net_expression():
    x = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 1],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneVectorCGPNet(net_params)
    print('get_expressions():')
    print(net.get_cgp_expressions())

    print('net ws:')
    print(net.get_ws())

    print('new get expressions:')
    exprs = pretty_net_exprs(net)
    print(exprs)


def test_net():
    x = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 2],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneLinearCGPNet(net_params)
    print(net.get_expressions())
    print(net.get_ws())
    results = net(x)
    [print(result, result.shape) for result in results]


def test_layer_torch():
    x = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float)

    var_names = ['x0', 'x1', 'x2']
    params = CGPParameters(1, 1,
                           5, 5, 101,
                           default_functions,
                           1)
    cgp_layer = OneExpCGPLayer(params)
    exprs = cgp_layer.get_expressions()
    print(exprs)
    print('CGP:', cgp_layer(x))


def test_net_torch():

    x = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 2],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneLinearCGPNet(net_params)
    _print_net(net, x)

    print('###encode it')
    encoded_net = OneLinearCGPNet.encode_net(net_params,
                                             genes_list=net.get_genes(),
                                             ephs_list=net.get_ephs(),
                                             w_list=net.get_ws(),
                                             bias_list=net.get_bias())
    _print_net(encoded_net, x)


def test_OneLinearCGPNet_OneExp():
    x = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 2],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneLinearCGPNet(net_params, clas_cgp=OneExpCGPLayer)
    _print_net(net, x)


if __name__ == '__main__':
    # for _ in range(5):
    # test_net()
    # test_layer_torch()
    # test_net_torch()
    test_net_expression()
    # test_OneLinearCGPNet_OneExp()