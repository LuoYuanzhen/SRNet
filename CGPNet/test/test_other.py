import torch


def test_torchmul():
    x = torch.normal(mean=0., std=1., size=(5, 1))
    w = torch.normal(mean=0., std=1., size=(5, 4))

    print(x)
    print(w)

    print(x * w)


if __name__ == '__main__':
    test_torchmul()
