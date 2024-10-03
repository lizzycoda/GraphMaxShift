import numpy as np

"""
This file contains code to obtain (an approximation of) the ground truth clustering as determiend by the gradient flow for a Gaussian mixture.

We use the method of Chacon which consists of:
1. Find the modes
2. Find the saddle points
3. Determine the boundaries 
"""

def ground_truth_clustering(mixture):
    """
    Apply the numerical method. Note in practice works well for bimodal Gaussian mixtures. For mixtures with more than 2 components, the method may require some additional adjustments. 
    """
    modes = find_modes(mixture)
    saddle = find_saddle_points(mixture)
    boundaries = get_boundaries(mixture, saddle)

    return modes, saddle, boundaries 


###### Determine the modes ######
def find_modes(mixture):  
    """
    Starting from the component means, use Newton's method to find a zero gradient point. 
    """
    
    modes = []
    for x0 in mixture.means:
        modes += [multivariable_newton(x0, f = mixture.grad, fprime = mixture.hessian, num_iter = 100)]
        
    return np.unique(modes, axis = 0).tolist() #uniqueness correction

def multivariable_newton(x0, f, fprime, num_iter):
    """
    Simple implementation of Newton's method to find a zero of f: R^d -> R. 
    This could be improved with a tolerance or other stopping criteria. 
    """
    
    xk = x0
    for i in range(num_iter):
        xk = xk - np.matmul(np.linalg.inv(fprime(xk)), f(xk))
    return xk


###### Determine the saddle points ######

def find_saddle_points(mixture):
    """
    Determine the saddle points by searching along the ridgeline between every pair of component means.
    
    Theorem 1 in Ray and Lindsay states that the saddle points must lie on the ridgeline surface
    (which is defined by all the component means). 
    
    Here, we only implement the simpler search method between pairs as described in Chacon. 
    
    """
    
    saddle = []
    K = mixture.K
    
    for i in range(K): #iterate over pairs, find 
        for j in range(i+1, K):
            ridgeline = get_ridgeline_func(mixture.means[i], mixture.means[j], mixture.cov[i], mixture.cov[j])

            grad_vals = []
            x_vals = []
            for alpha in np.arange(0.1, .91, .01): #want to exclude points too close to means
                x_alpha = ridgeline(alpha)
                grad_vals += [np.linalg.norm(mixture.grad(x_alpha))]
                x_vals += [x_alpha]

            m = np.argmin(grad_vals)
            saddle += [x_vals[m]]
    return saddle
        


def get_ridgeline_func(mu1, mu2, cov1, cov2):
    """
    Get the ridgeline function as defined in Ray & Lindsay(2015) for a pair of parameters
    """
    
    C1 = np.linalg.inv(cov1)
    C2 =np.linalg.inv(cov2)
    v1 = np.matmul(C1,mu1)
    v2 = np.matmul(C2,mu2)
    
    def r(alpha):
        beta = 1-alpha
        M1 = alpha*C1 + beta*C2 #could st
        M2 = np.linalg.inv(M1)
        u1 = alpha*v1 + beta*v2
        return np.matmul(M2, u1)
    return r



###### Determine the boundaries ######

def get_boundaries(mixture, saddle_pts, perturbation = .01, num_steps = 5000, step_size = 1, scale_factor = 10e4):
    
    """
    For each saddle point, obtain the boundaries by solving the IVP y'(t) = Df(y(t)) via Euler's method where f is the mixture density. 
    Start from a point slightly shifted from the saddle point in the direction of the eigenvector of the Hessian corresponding to a negative eigenvalue. 
    
    Inputs:
    mixture: A GaussianMixture object
    saddle_pts: List of saddle points
    perturbation: shift from saddle point to initialize the IVP 
    num_steps: parameter in Euler's method
    step_size: parameter un Euler's method
    scale_factor: used in Euler's method to avoid getting stuck 
    
    Returns: A list of boundaries 
    """
    
    boundaries = []
    for saddle in saddle_pts:
        H = mixture.hessian(saddle)
        eig, V = np.linalg.eigh(H) #first column should correspond to negative eigenvalue
        
        # boundaries between population clusters are the unstable manifolds of the saddle points 
        # shift by +/- pertubation to get the full boundary 
        y0 = V[:,0]*perturbation + saddle 
        bd1 = euler(y0, mixture.grad, step_size, num_steps, scale_factor) # step size needs to be large because gradient is small  
        y0 = -V[:,0]*perturbation + saddle 
        bd2 = euler(y0, mixture.grad, step_size, num_steps, scale_factor)        
        
        boundaries += [bd1, bd2]
    return boundaries

    
    
def euler(y0, f, step_size, num_iter = 1000, scale_factor = 10e4): 
    """
    Applies num_iter/2 iterations of Euler's method with step_size to solve the IVP y'(t) = f(t, y(t)) with y(0) = y0.
    Then, the step size is increases by scale_factor and  num_iter/2 more iterations are performed.
    """
    
    path = [y0]
    y = y0
    for i in range(int(num_iter/2)):
        y = y - step_size*f(y)
        path += [y]
    step_size = step_size*scale_factor # increase step size for good numerical behavior 
    for i in range(int(num_iter/2)):
        y = y - step_size*f(y) 
        path += [y]
    return np.array(path)
