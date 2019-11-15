import numpy as np
class algorithm:
    def __init__(self, c = 0.0005, eta = 0.002, max_iter = 1000, stop_citerier = 1e-3):
        self.c = c
        self.eta = eta
        self.max_iter = max_iter
        self.stop_citerier = stop_citerier

    def fit(self, x, t):
        n = len(x[0])
        m = len(x)

        #initial variables
        self.w = np.zeros(n)
        self.b = 0.0
        y = np.dot(x, self.w) + self.b

        #iterative step
        for i in range(self.max_iter):
            dw = -2*np.matmul( x.T, (t-y))/m + 2*self.w*self.c
            self.w += dw*self.eta # w = w + dw*self.eta
            db = -2*((t-y)).sum()/m
            self.b += db*self.eta
            y = np.dot(x, self.w) + self.b
            if np.sqrt(np.dot(dw,dw) + db**2) <= self.stop_citerier:
                break

        return self

    def predict(self, z):
        return np.dot(z, self.w) + self.b












'''
class algorithm:
    def __init__(self, c = 1, eta = 0.0002, max_iter = 10000, stop_citerier = 1e-3):
        self.c = c
        self.eta = eta
        self.max_iter = max_iter
        self.stop_citerier = stop_citerier
    
    #Learning
    def fit(self,x,t):
        n = len(x[0])
        m = len(x)
        #initial variables
        self.w = np.zeros(n)
        self.b = 1.0
        y = np.dot(x, self.w) + self.b

        #iterative step
        for i in range(self.max_iter):
            self.eta = self.eta/(i**2+1)
            dw = -self.c*np.dot((t-y), x)/m + self.w
            self.w += dw*self.eta
            db = -self.c*((t-y)).sum()/m + self.b
            self.b += db*self.eta
            h = np.dot(x, self.w) + self.b 
            if np.sqrt(np.dot(dw,dw) + db**2) <= self.stop_citerier:
                break

        return self

    #prediction
    def predict(self, z):
        return np.dot(z, self.w) + self.b

'''