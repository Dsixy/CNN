class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.t = 0

    def update(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient * gradient
        self.m_hat = self.m / (1 - self.beta1 ** self.t)
        self.v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.m_hat / (self.v_hat ** 0.5 + self.epislon)