class config(object):
    def __init__(self):
        self.batch_size = 12
        self.epochs = 150
        self.learning_rate = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-08
        self.weight_decay = 0
        self.amshrad = False
        self.maximize = False
        self.dice_smooth = 1e-5

