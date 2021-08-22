from dCGPNet.layers import OneExpOneOutCGPLayer, MulExpCGPLayer, OneExpCGPLayer
from dCGPNet.methods import LBFGSTrainer, SGDTrainer, PSOTrainer, NewtonTrainer
from dCGPNet.nets import OneVectorCGPNet, OneLinearCGPNet, DoubleLinearCGPNet


clas_net_map = {
    'OneVectorCGPNet': OneVectorCGPNet,
    'OneLinearCGPNet': OneLinearCGPNet,
    'DoubleLinearCGPNet': DoubleLinearCGPNet
}


clas_cgp_map = {
    'OneExpOneOutCGPLayer': OneExpOneOutCGPLayer,
    'OneExpCGPLayer': OneExpCGPLayer,
    'MulExpCGPLayer': MulExpCGPLayer
}


clas_optim_map = {
    'SGD': SGDTrainer,
    'Newton': NewtonTrainer,
    'LBFGS': LBFGSTrainer,
    'PSO': PSOTrainer
}






