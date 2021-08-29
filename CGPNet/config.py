from CGPNet.layers import OneExpOneOutCGPLayer, MulExpCGPLayer, OneExpCGPLayer
from CGPNet.methods import LBFGSTrainer, SGDTrainer, PSOTrainer, NewtonTrainer
from CGPNet.nets import OneVectorCGPNet, OneLinearCGPNet, DoubleLinearCGPNet


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






