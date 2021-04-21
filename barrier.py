import numpy as np
from problems import CenteringProblem

def backtracking_search(centering_problem, v, search_dir, grad, alpha=0.3, beta=0.9):
    '''Return the best step to update the solution.'''
    
    # Initialize
    step = 1
    f = centering_problem.compute_obj(v)
    f_new = centering_problem.compute_obj(v+step*search_dir)
    
    # Find step
    while centering_problem.is_feasible(v) and f_new > f + alpha*step*grad.T.dot(search_dir).item():
        f_new = centering_problem.compute_obj(v+step*search_dir)
        step *= beta
        v += step*search_dir
        if step < 1e-5:
            # print('forced break backtracking_search')
            break
    
    return step


def centering_step(Q, p, A, b, t, v0, eps):
    '''
    Minimizes f with the Newton's method, with initial point v0 and precision eps.

    Returns:
       v: array
           list of points v that increasingly minimize f
    '''
    Vs = []
    centering_problem = CenteringProblem(Q, p, A, b, t)
    v = v0.copy()
    
    while True:
        Vs.append(v)

        # Retrieve elements
        grad = centering_problem.compute_grad(v)
        hessian = centering_problem.compute_hessian(v)
        
        # Get parameters to decrease the function
        search_dir = -np.linalg.solve(hessian, grad)
        lambda_2 = -np.dot(grad.T, search_dir).item()

        # Check stopping criterion
        if lambda_2 < 2*eps:
            break
            
        step = backtracking_search(centering_problem, v, search_dir, grad)
        v += step*search_dir

    return Vs


def barr_method(Q, p, A, b, v0, eps=1e-6, eps_centering=1e-6, t=1, mu=5):
    '''
    Minimizes the constrained function with the Barrier's method, with initial point v0 and precision eps 
    (and precision for the Newton's method eps_centering).

    Returns:
       Vs: array
           points v that increasingly minimize the constrained function.
       Ts: array
           to keep track of the duality gap
       Cs: array
           number of steps per Centering iteration
    '''
    # Initialiaze
    v = v0.copy()
    m = len(b)
    Vs, Ts, Cs = [v], [t], [0]
    
    while m/t > eps:
        
        # Centering step and retrieve the result
        v_list = centering_step(Q, p, A, b, t, v, eps_centering)
        v = v_list[-1]
        t *= mu
        
        # Save the important informations
        Vs.append(v)
        Ts.append(t)
        Cs.append(Cs[-1]+len(v_list)-1)

    return Vs, Ts, Cs