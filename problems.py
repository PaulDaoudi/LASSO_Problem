import numpy as np

# Help class to compute the required elements

class CenteringProblem:
    '''
    Class computing the required elements for the newton method.
    Can also compute the original objective if required.
    '''
    
    def __init__(self, Q, p, A, b, t):
        self.Q = Q
        self.p = p
        self.A = A
        self.b = b
        self.t = t
        
    def compute_init_obj(self, v):
        return np.dot(np.dot(v.T, self.Q), v).item() + self.p.T.dot(v).item()
        
    def compute_obj(self, v):
        barrier = -np.log(self.b-self.A.dot(v)).sum()
        return self.t*self.compute_init_obj(v) + barrier
    
    def compute_grad(self, v):
        z = 1/(self.b - self.A.dot(v))
        grad = self.t*np.dot(self.Q.T + self.Q, v) + self.t*self.p + np.dot(self.A.T, z)
        return grad
    
    def compute_hessian(self, v):
        z = 1/(self.b - self.A.dot(v))
        hessian = self.t*(self.Q.T + self.Q) + np.dot(self.A.T.dot(np.diag(z.reshape(-1))**2),self.A)
        return hessian
    
    def is_feasible(self, v):
        '''
        The point computed in the backtracking_search can go outside the domain of the function (in here log).
        We need to check it is not the case.
        '''
        return np.all((self.b - self.A.dot(v)) > 0)