import torch
from torch import nn

from dCGPNet.layers import BaseCGP, OneExpOneOutCGPLayer, LinearLayer, MulExpCGPLayer, OneExpCGPLayer
from dCGPNet.params import NetParameters, CGPParameters


net_cgp_dims = {
    BaseCGP.__name__: (None, None),
    OneExpOneOutCGPLayer.__name__: (None, 1),
    MulExpCGPLayer.__name__: (1, None),
    OneExpCGPLayer.__name__: (1, 1)
}


def _create_cgp_layer(clas_cgp, net_params, idx_neurons, genes=None, ephs=None):
    dims = net_cgp_dims[clas_cgp.__name__]
    n_inputs = dims[0] if dims[0] else net_params.neurons[idx_neurons-1]
    n_outputs = dims[1] if dims[1] else net_params.neurons[idx_neurons]

    cgp_params = CGPParameters(n_inputs=n_inputs,
                               n_outputs=n_outputs,
                               n_rows=net_params.n_rows,
                               n_cols=net_params.n_cols,
                               levels_back=net_params.levels_back,
                               function_set=net_params.function_set,
                               n_eph=net_params.n_eph)
    return clas_cgp(cgp_params, genes, ephs)


class BaseCGPNet(nn.Module):
    def __init__(self, net_params, cgp_layers=None, clas_cgp=BaseCGP):
        super(BaseCGPNet, self).__init__()
        self.net_params = net_params

        self.clas_cgp = clas_cgp
        self.neurons = net_params.neurons
        self.n_layer = len(self.neurons)  # n_layer includes input layer, hidden layers, output layer.
        self.fitness = None
        self.fitness_list = []

        self.cgp_layers = []
        if cgp_layers:
            self.cgp_layers = cgp_layers
        else:
            for i in range(1, self.n_layer):
                self.cgp_layers.append(_create_cgp_layer(self.clas_cgp, self.net_params, i))

    def get_cgp_expressions(self):
        """Note that this method only return sympy expressions of each layer, not including w"""
        exprs = []
        for layer in self.cgp_layers:
            exprs.append(layer.get_expressions())
        return exprs

    def get_genes(self):
        """get a list of genes in CGPLayer"""
        genes_list = []
        for layer in self.cgp_layers:
            genes_list.append(layer.get_genes())
        return genes_list

    def get_ephs(self):
        """get a list of ephs in CGPLayer"""
        ephs_list = []
        for layer in self.cgp_layers:
            ephs_list.append(layer.get_ephs())
        return ephs_list


class OneVectorCGPNet(BaseCGPNet):
    def __init__(self, net_params, cgp_layers=None, nn_layers=None, clas_cgp=OneExpOneOutCGPLayer):
        super(OneVectorCGPNet, self).__init__(net_params, cgp_layers, clas_cgp=clas_cgp)

        self.add_bias = self.net_params.add_bias
        self.nn_layers = []
        if nn_layers:
            self.nn_layers = nn_layers
        else:
            for i in range(1, self.n_layer):
                self.nn_layers.append(LinearLayer(1, self.neurons[i], add_bias=self.add_bias))

    def __call__(self, x):
        layer_output = x
        outputs = []

        for i in range(0, self.n_layer - 1):
            layer_output = self.nn_layers[i](self.cgp_layers[i](layer_output))
            outputs.append(layer_output)

        return outputs

    def get_ws(self):
        """get all the weights in nn_layers, and make them detach, shape: (n_input, n_output)"""
        w_list = []
        for layer in self.nn_layers:
            w_list.append(layer.get_weight())
        return w_list

    def get_bias(self):
        """get all the biases in nn_layers, and make them detach, shape: (n_output, 1)"""
        bias_list = []
        if self.add_bias:
            for layer in self.nn_layers:
                bias_list.append(layer.get_bias().view(1, -1))
        return bias_list

    def get_net_parameters(self):
        parameters = []
        for layer in self.nn_layers:
            parameters += list(layer.parameters())
        return parameters

    @classmethod
    def encode_net(cls,
                   net_params: NetParameters,
                   genes_list, ephs_list, w_list, bias_list=None, clas_cgp=MulExpCGPLayer):
        """class method that encode the net based on existing params"""
        neurons = net_params.neurons
        n_layers = len(neurons)

        add_bias = bias_list is not None
        if net_params.add_bias != add_bias:
            raise ValueError(f'net_params.add_bias is {net_params.add_bias} while bias_list is {bias_list}')
        if len(genes_list) != len(ephs_list) != len(w_list) and n_layers != len(genes_list) + 1:
            raise ValueError('length of genes, ephs, W should all be equal to n_layer - 1!')
        if add_bias and len(bias_list) != len(w_list):
            raise ValueError('length of bias, genes, ephs, W should all be eqaul to n_layer - 1!')

        cgp_layers, nn_layers = [], []
        for i in range(1, n_layers):
            genes, ephs = genes_list[i - 1], torch.tensor(ephs_list[i - 1])
            W = torch.tensor(w_list[i - 1])
            b = torch.tensor(bias_list[i-1]) if add_bias else None

            cgp_layers.append(_create_cgp_layer(clas_cgp, net_params, i, genes, ephs))
            nn_layers.append(LinearLayer(weight=W, bias=b, add_bias=add_bias))

        return cls(net_params, cgp_layers, nn_layers, clas_cgp=clas_cgp)

    def generate_offspring(self, gidxs_list, mutant_genes_list):
        cgp_layers, nn_layers = [], []
        for n, cgp, gidxs, mutant_genes in zip(self.nn_layers, self.cgp_layers, gidxs_list, mutant_genes_list):
            cgp_layers.append(cgp.generate_offspring(gidxs, mutant_genes))
            nn_layers.append(n.clone())
        return OneVectorCGPNet(self.net_params, cgp_layers, nn_layers)


class OneLinearCGPNet(BaseCGPNet):
    """Using MulExpCGPLayer as CGP layer.
    Each layer's output is like: [f0(x*w^T), f1(x*w^T), ...]"""
    def __init__(self, net_params, cgp_layers=None, nn_layers=None, add_bias=False, clas_cgp=MulExpCGPLayer):
        super(OneLinearCGPNet, self).__init__(net_params, cgp_layers, clas_cgp=clas_cgp)

        self.add_bias = add_bias
        self.nn_layers = []
        if nn_layers:
            self.nn_layers = nn_layers
        else:
            for i in range(1, self.n_layer):
                self.nn_layers.append(LinearLayer(self.neurons[i - 1], self.neurons[i], add_bias=self.add_bias))

    def __call__(self, x):
        layer_output = self.cgp_layers[0](self.nn_layers[0](x))
        outputs = [layer_output]

        for i in range(1, self.n_layer-1):
            layer_output = self.cgp_layers[i](self.nn_layers[i](layer_output))
            outputs.append(layer_output)

        return outputs

    def set_ws(self, w_list):
        if len(w_list) != self.n_layer - 1:
            raise ValueError(f"expected w_list'length {self.n_layer-1}, but got {len(w_list)}")
        for i, w in enumerate(w_list):
            self.nn_layers[i].set_weight(w)

    def get_ws(self):
        """get all the weights in nn_layers, and make them detach, shape: (n_input, n_output)"""
        w_list = []
        for layer in self.nn_layers:
            w_list.append(layer.get_weight())
        return w_list

    def get_bias(self):
        """get all the biases in nn_layers, and make them detach, shape: (n_output, 1)"""
        bias_list = []
        if self.add_bias:
            for layer in self.nn_layers:
                bias_list.append(layer.get_bias().view(1, -1))
        return bias_list

    def get_net_parameters(self):
        parameters = []
        for layer in self.nn_layers:
            parameters += list(layer.parameters())
        return parameters

    @classmethod
    def encode_net(cls,
                   net_params: NetParameters,
                   genes_list, ephs_list, w_list, bias_list=None, clas_cgp=MulExpCGPLayer):
        """class method that encode the net based on existing params"""
        neurons = net_params.neurons
        n_layers = len(neurons)

        if len(genes_list) != len(ephs_list) != len(w_list) and n_layers != len(genes_list) + 1:
            raise ValueError('length of genes, ephs, W should all be equal to n_layer - 1!')

        cgp_layers, nn_layers = [], []
        for i in range(1, n_layers):
            genes, ephs, W = genes_list[i - 1], torch.tensor(ephs_list[i - 1]), torch.tensor(w_list[i - 1])
            bias = bias_list[i - 1] if bias_list else None

            cgp_layers.append(_create_cgp_layer(clas_cgp, net_params, i, genes, ephs))
            nn_layers.append(LinearLayer(weight=W, bias=bias))

        return cls(net_params, cgp_layers, nn_layers, clas_cgp=clas_cgp)

    def generate_offspring(self, gidxs_list, mutant_genes_list):
        cgp_layers, nn_layers = [], []
        for n, cgp, gidxs, mutant_genes in zip(self.nn_layers, self.cgp_layers, gidxs_list, mutant_genes_list):
            cgp_layers.append(cgp.generate_offspring(gidxs, mutant_genes))
            nn_layers.append(n.clone())
        return OneLinearCGPNet(self.net_params, cgp_layers, nn_layers)


class DoubleLinearCGPNet(BaseCGPNet):
    def __init__(self, net_params, cgp_layers=None, matrix_layers=None, vector_layers=None, add_bias=False, clas_cgp=OneExpOneOutCGPLayer):
        super(DoubleLinearCGPNet, self).__init__(net_params, cgp_layers, clas_cgp=clas_cgp)

        self.add_bias = add_bias

        self.nn_layers = []
        if matrix_layers and vector_layers:
            self.matrix_layers = matrix_layers
            self.vector_layers = vector_layers
        else:
            for i in range(1, self.n_layer):
                self.matrix_layers.append(LinearLayer(self.neurons[i - 1], self.neurons[i], add_bias=self.add_bias))
                self.vector_layers.append(LinearLayer(1, self.neurons[i], add_bias=self.add_bias))

    def __call__(self, x):
        layer_output = x
        outputs = []
        for m_layer, v_layer, cgp_layer in zip(self.matrix_layers, self.vector_layers, self.cgp_layers):
            layer_output = v_layer(cgp_layer(m_layer(layer_output)))
            outputs.append(layer_output)
        return outputs

    def get_ws(self):
        m_list, v_list = [], []
        for m_layer, v_layer in zip(self.matrix_layers, self.vector_layers):
            m_list.append(m_layer.get_weight())
            v_layer.append(v_layer.get_weight())
        return m_list, v_list

    def get_net_parameters(self):
        parameters = []
        for m_layer, v_layer in zip(self.matrix_layers, self.vector_layers):
            parameters += list(m_layer.parameters()) + list(v_layer.parameters())
        return parameters

    @classmethod
    def encode_net(cls,
                   net_params: NetParameters,
                   genes_list, ephs_list, w_list_turple, bias_list=None, clas_cgp=OneExpOneOutCGPLayer):
        """class method that encode the net based on existing params"""
        neurons = net_params.neurons
        n_layers = len(neurons)

        if len(genes_list) != len(ephs_list) != len(w_list_turple[0]) != len(w_list_turple[1]) and n_layers != len(genes_list) + 1:
            raise ValueError('length of genes, ephs, W should all be equal to n_layer - 1!')

        cgp_layers, m_layers, v_layers = [], [], []
        for i in range(1, n_layers):
            genes, ephs = genes_list[i - 1], torch.tensor(ephs_list[i - 1])
            m_weight, v_weight = torch.tensor(w_list_turple[0][i - 1], w_list_turple[1][i-1])
            bias = bias_list[i - 1] if bias_list else None

            cgp_layers.append(_create_cgp_layer(clas_cgp, net_params, i, genes, ephs))
            m_layers.append(LinearLayer(weight=m_weight))
            v_layers.append(LinearLayer(weight=v_weight, bias=bias))

        return cls(net_params, cgp_layers, m_layers, v_layers, clas_cgp=clas_cgp)

    def generate_offspring(self, gidxs_list, mutant_genes_list):
        cgp_layers, m_layers, v_layers = [], [], []
        for m_layer, v_layer, cgp, gidxs, mutant_genes in zip(self.matrix_layers, self.vector_layers, self.cgp_layers, gidxs_list, mutant_genes_list):
            cgp_layers.append(cgp.generate_offspring(gidxs, mutant_genes))
            m_layers.append(m_layer.clone())
            v_layers.append(v_layer.clone())
        return DoubleLinearCGPNet(self.net_params, cgp_layers, m_layers, v_layers)



