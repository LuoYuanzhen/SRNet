import torch

from dataset_config import FUNC_MAP, ELITE_MAP, VALID_MAP, INTER_MAP
from data_utils import io
from exp_utils import encode_individual_from_json, draw_hidden_heat_compare_img, draw_decision_bound, \
    draw_output_compare_curves, draw_project_output_scatter
from neural_networks.nn_models import NN_MAP


def _encode_individual(json_file, which):
    """higher level encode method"""
    individual = encode_individual_from_json(json_file, which)
    return individual


def hidden_heat_map():
    dir = '../dataset/'
    filename = 'kkk0'

    json_file = f'/home/luoyuanzhen/cgpnet_result/log_v2/{filename}_30log.json'
    img_dir = f'/home/luoyuanzhen/cgpnet_result/test/'
    which = 'elite[0]'

    nn_dir = f"{dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{dir}{filename}')
    x_inner = true_inner[:, :-1]

    individual = _encode_individual(json_file, which)

    srnn_layer_inner = individual(x_inner)
    nn_layer_inner = io.get_nn_datalist(nn_dir)[1:]

    print(len(srnn_layer_inner), len(nn_layer_inner))
    for i in range(len(srnn_layer_inner)):
        srnn_layer_inner[i] = srnn_layer_inner[i].detach()

    draw_hidden_heat_compare_img(f'{img_dir}{filename}_{which}',
                                 nn_layer_inner[:-1],
                                 srnn_layer_inner[:-1],
                                 title=filename)


def output_curves():
    """draw the output curves, note that every variable x woule be the same as x[0]"""
    dir = '../dataset/'
    filename = 'kkk0'
    json_file = f'../cgpnet_result/logs/{filename}_30log.json'
    img_dir = '../cgpnet_result/imgs/'

    nn_dir = f"{dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{dir}{filename}')
    n_var = true_inner.shape[1] - 1

    x0_range = VALID_MAP[filename][0]
    inter_range = INTER_MAP[filename][0]
    x = torch.linspace(x0_range[0], x0_range[1], 1000).unsqueeze(1)
    x = x.repeat(1, n_var)
    x, _ = torch.sort(x, dim=0)

    func = FUNC_MAP[filename]
    true_output = func(*[x[:, i] for i in range(n_var)])

    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    nn_output = nn(x)[-1].detach()

    which = f'elite[0]'
    individual = _encode_individual(json_file, which)
    srnn_output = individual(x)[-1].detach()
    ys = [true_output, nn_output, srnn_output]
    labels = ['True', 'MLP', 'CGPNet']
    draw_output_compare_curves(x[:, 0], ys, labels,
                               inter_range=inter_range,
                               n_var=n_var,
                               title=f'{filename}',
                               savepath=f'{img_dir}{filename}_curves_{which}.pdf')


def output_curves_interpolate():
    dir = '../dataset/'
    filename = 'kkk0'
    json_file = f'../cgpnet_result/logs/{filename}_30log.json'
    img_dir = '../cgpnet_result/imgs/'

    nn_dir = f"{dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{dir}{filename}')
    n_var = true_inner.shape[1] - 1

    x0_range = INTER_MAP[filename][0]
    x = torch.linspace(x0_range[0], x0_range[1], 1000).unsqueeze(1)
    x = x.repeat(1, n_var)
    x, _ = torch.sort(x, dim=0)

    func = FUNC_MAP[filename]
    true_output = func(*[x[:, i] for i in range(n_var)])

    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    nn_output = nn(x)[-1].detach()

    which = f'elite[0]'
    individual = _encode_individual(json_file, which)
    srnn_output = individual(x)[-1].detach()
    ys = [true_output, nn_output, srnn_output]
    labels = ['True', 'MLP', 'CGPNet']
    draw_output_compare_curves(x[:, 0], ys, labels,
                               n_var=n_var)


def project_output_scatter():
    dir = '../dataset/'
    filename = 'kkk0'
    json_file = f'../cgpnet_result/logs/{filename}_30log.json'
    img_dir = '../cgpnet_result/imgs/'

    which = 'elite[0]'
    nn_dir = f"{dir}{filename}_nn/"

    x_inner = io.get_dataset(f'{dir}{filename}_nn/input')

    x_range, inter_ranges = VALID_MAP[filename], INTER_MAP[filename]
    n_sample = x_inner.shape[0]
    x = []
    for i, ranges in enumerate(zip(x_range, inter_ranges)):
        inter, extra = ranges
        xi_left = (inter[0] - extra[0]) * torch.rand(n_sample//2, 1) + extra[0]
        xi_right = (extra[1] - inter[1]) * torch.rand(n_sample//2, 1) + inter[1]
        xi_inner = x_inner[:, i].unsqueeze(1)
        x.append(torch.vstack((xi_left, xi_inner, xi_right)))
    x = torch.hstack(x)

    n_var = x.shape[1]

    individual = _encode_individual(json_file, which)
    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    func = FUNC_MAP[filename]

    srnn_output = individual(x)[-1].detach()
    true_output = func(*[x[:, i] for i in range(n_var)])
    nn_output = nn(x)[-1].detach()

    draw_project_output_scatter(x, [true_output, nn_output, srnn_output], ['True', 'MLP', 'CGPNet'], inter_ranges)


def project_output_scatter_interpolate():
    dir = '../dataset/'
    filename = 'kkk1'
    json_file = f'/home/luoyuanzhen/cgpnet_result/log_extra/{filename}_30log.json'
    img_dir = '/home/luoyuanzhen/cgpnet_result/img/'

    nn_dir = f"{dir}{filename}_nn/"

    x = io.get_dataset(f'{dir}{filename}_nn/input')

    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    nn_output = nn(x)[-1].detach()

    true_output = FUNC_MAP[filename](*list([x[:, i] for i in range(x.shape[1])]))

    vars = tuple(list([x[:, i] for i in range(x.shape[1])]))

    # top 10
    for i in range(10):
        which = f'elite[{i}]'
        individual = _encode_individual(json_file, which)
        srnn_output = individual(x)[-1].detach()

        ys = [nn_output, true_output, srnn_output]
        labels = ['True', 'MLP', 'CGPNet']
        draw_project_output_scatter(x, ys, labels)


def classifier_decisions_bounds():
    dir = '/home/luoyuanzhen/Datasets/regression/feynman/'
    filename = 'feynman2'

    json_file = f'/home/luoyuanzhen/cgpnet_result/single_log_v3/{filename}_30log.json'
    which = 'elite[0]'

    individual = encode_individual_from_json(json_file, which)
    draw_decision_bound(dir, filename, individual, from_pmlb=False)


def test():
    import sympy as sp

    m1, m2, r1 = sp.symbols('m1 m2 r1')
    exp = 53.16*sp.cos(0.27*sp.log(0.015*r1/(m2 - 0.21))/(sp.log(0.015*r1/(m2 - 0.21)) + sp.cos(0.72*(m2 - 0.21)/r1))) - 41.11*sp.log(0.015*r1/(m2 - 0.21))/(sp.log(0.015*r1/(m2 - 0.21)) + sp.cos(0.72*(m2 - 0.21)/r1))
    # str = '53.16*cos(0.27*log(0.015*r1/(m2 - 0.21))/(log(0.015*r1/(m2 - 0.21)) + cos(0.72*(m2 - 0.21)/r1))) - 41.11*log(0.015*r1/(m2 - 0.21))/(log(0.015*r1/(m2 - 0.21)) + cos(0.72*(m2 - 0.21)/r1))'
    print(sp.pprint(exp))
    print(sp.latex(exp))


if __name__ == '__main__':
    # hidden_heat_map()
    output_curves()
    # output_curves_interpolate()
    # project_output_scatter()
    # project_output_scatter_interpolate()
    # test()












