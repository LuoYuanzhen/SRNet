class CGPNet:
    def __init__(self, net_params, f_cgps=None, w_cgps=None):
        self.net_params = net_params

        self.neurons = net_params.neurons
        self.n_layer = len(self.neurons)

        self.fitness = None
        self.fitness_list = []

        self.f_cgps = []
        self.w_cgps = []
        if f_cgps and w_cgps:
            self.f_cgps = f_cgps
            self.w_cgps = w_cgps
        else:
            for i in range(self.n_layer):
                pass