# effective_complexity/training/early_stopping.py

class EarlyStopping:
    """
    Stops when validation loss stops improving.
    """

    def __init__(self, patience=15, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.wait = 0
        self.best_epoch = 0

    def step(self, value, epoch):
        if self.best - value > self.delta:
            self.best = value
            self.wait = 0
            self.best_epoch = epoch
            return False, True   # stop?, improved?

        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True, False
            return False, False
