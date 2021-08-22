import numpy as np
from data_utils import io


def kkk(dir):
    # kkk0
    x = np.random.uniform(1, 1.5, (100, 1))
    y = np.sin(x) + np.sin(x+x**2)
    kkk0 = np.hstack((x, y))
    io.save_parameters(kkk0, f'{dir}kkk/kkk0_outer')

    # kkk1
    x0 = np.random.uniform(1, 1.5, (100, 1))
    x1 = np.random.uniform(1, 1.5, (100, 1))
    y = 2 * np.sin(x0) * np.cos(x1)
    kkk1 = np.hstack((x0, x1, y))
    io.save_parameters(kkk1, f'{dir}kkk/kkk1_outer')

    # kkk2
    x = np.random.uniform(50, 75, (100, 1))
    y = 3 + 2.13 * (np.log(x) / np.log(np.e))
    kkk2 = np.hstack((x, y))
    io.save_parameters(kkk2, f'{dir}kkk/kkk2_outer')

    # kkk3
    x0 = np.random.uniform(5, 7.5, (10, 1))
    x1 = np.random.uniform(5, 7.5, (10, 1))
    y = 1 / (1 + x0**(-4) + 1 / x1**(-4))
    kkk3 = np.hstack((x0, x1, y))
    io.save_parameters(kkk3, f'{dir}kkk/kkk3_outer')

    # kkk4
    x0 = np.random.uniform(1, 1.5, (500, 1))
    x1 = np.random.uniform(1, 1.5, (500, 1))
    x2 = np.random.uniform(2, 2.5, (500, 1))
    y = 30 * x0 * x1 / ((x0 - 10) * x2 * x2)
    kkk4 = np.hstack((x0, x1, x2, y))
    io.save_parameters(kkk4, f'{dir}kkk/kkk4_outer')

    # kkk5
    x0 = np.random.uniform(3, 4.5, (10, 1))
    x1 = np.random.uniform(3, 4.5, (10, 1))
    y = x0 * x1 + np.sin((x0 - 1) * (x1 - 1))
    kkk5 = np.hstack((x0, x1, y))
    io.save_parameters(kkk5, f'{dir}kkk/kkk5_outer')

    print(kkk0.shape, kkk1.shape, kkk2.shape, kkk3.shape, kkk4.shape, kkk5.shape)


if __name__ == '__main__':
    data_dir = '/home/luoyuanzhen/Datasets/regression/'
    kkk(data_dir)
