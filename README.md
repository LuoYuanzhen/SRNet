# 1. Introduction
A Symbolic Regression (SR) method called SRNet to mine hidden semantics of each network layer in Neural Network 
(Multi-Layer Perceptron). SRNet is an evolutionary computing algorithm, leveraging Multi-Chromosomes Cartesian Genetic 
Programming (MCCGP) to find mathmatical formulas $f_i(x)*w_i+b_i$ for each network layer, white-boxing the black box.
# 2. Run
Simply run cgp_net_experiments.py to run 12 benchmarks in 'dataset/' 30 times. Then run analyse/best_draw.py to get all 
the experiment figures. 