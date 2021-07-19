class EMA():
    """
    Exponential moving average
    This has been borrowed from https://github.com/galsang/BiDAF-pytorch
    """

    def __init__(self, mu: float, numerator: float = 1.0, denominator: float = 5.0):
        self.mu = mu
        self.shadow = {}
        self.numerator = numerator
        self.denominator = denominator

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x, num_updates=None):
        if num_updates is not None:
            mu = min(self.mu,
                        (self.numerator + num_updates) / (self.denominator + num_updates))
        else:
            mu = self.mu

        assert name in self.shadow
        new_average = (1.0 - mu) * x + mu * self.shadow[name]
        self.shadow[name] = new_average.clone()

    @staticmethod
    def ema_restore_backed_params(backup_params, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    @staticmethod
    def ema_backup_and_loadavg(ema, model):
        backup_params = EMA(0)
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup_params.register(name, param.data)
                param.data.copy_(ema.get(name))
        return backup_params

    @staticmethod
    def ema_register(config, model):
        ema = EMA(config["exp_decay_rate"])
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
        return ema

    @staticmethod
    def ema_update(ema, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)