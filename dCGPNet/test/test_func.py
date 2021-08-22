import sympy as sp
import torch

from dCGPNet.functions import function_map


def test_function():
    vars = [sp.Symbol('x0'), sp.Symbol('x1')]
    for str_func in function_map:
        func = function_map[str_func]
        if func.arity == 1:
            print(func(vars[0], is_pt=False))
        else:
            print(func(*vars, is_pt=False))

    vals = [torch.tensor([0.2], dtype=torch.float), torch.tensor([0.1], dtype=torch.float)]
    for str_func in function_map:
        func = function_map[str_func]
        if func.arity == 1:
            print(func(vals[0]))
        else:
            print(func(*vals))


def test_sp():
    vars = [sp.Symbol('x0'), 1.2]
    # variable = sp.Matrix(vars).T
    # print(variable)
    exp = vars[0] * 2 + 1
    print(exp)
    print(exp.subs(sp.Symbol('x0'), [[1, 2, 3]]))


if __name__ == '__main__':
    # test_function()
    test_sp()
