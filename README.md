# 1. Introduction
A Symbolic Regression (SR) method called SRNet to mine hidden semantics of each network layer in Neural Network 
(Multi-Layer Perceptron). SRNet is an evolutionary computing algorithm, leveraging Multi-Chromosomes Cartesian Genetic 
Programming (MCCGP) to find mathmatical formulas $f_i(x)*w_i+b_i$ for each network layer, white-boxing the black box.
# 2. How to Run
## Training SRNet 
Simply run cgp_net_experiments.py to run 12 benchmarks in 'dataset/' 30 times. The selected elites in all independent
runs would be stored at cgpnet_result/ as json file; the convergence curve of fitness would be saved at
cgpnet_result/ as pdf file.
## Get experimental Figures
After running the cgp_net_experiments.py, we can get all the experimental figures. 
### Interpolation/Extrapolation Curves/Points and Heat Map
Simply run analyse/best_draw.py to get the experiment figures, including the fitting curves (or distribution points)
of SRNet in interpoaltion/extrapolation domain, the heat map of comparition of outputs of SRNet layer vs the NN layer.

For example, the fitting curves (or distribution points) of SRNet in interpolation/extrapolation domain of K0:
![The fitting curves of SRNet in extrapolation domain of K0](https://github.com/LuoYuanzhen/SRNet/blob/master/IMG/kkk0_curves_elite%5B0%5D.png)
The heat map of comparition of outputs of SRNet layer vs the NN layer:

### Compare to LIME and MAPLE on Extrapolation Domain
Simply run compare_experimetns.py to get the best individual SRNet in 30 times runing vs LIME vs MAPLE on the interpolation
and extrapolation domain:

