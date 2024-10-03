import numpy as np
import scipy.stats as stats

class GaussianMixture:

    def __init__(self,weights, means, cov):
        
        """
        N-dimensional Gaussian mixture with parameters:
        weights: a list of K mixture weights that sum to 1
        means: a list of K (Nx1) vectors, each of which is the mean of a mixture component
        cov: a list of K (NxN) arrays, each of which is the covariance matrix a mixture component
    
        """
        self.K = len(weights)
        self.weights = weights
        self.means = means
        self.cov = cov
        self.inv_cov = [np.linalg.inv(x) for x in self.cov]
        

        
    def sample(self,n):
        """ Sample n points from the Gaussian mixture. 
        Returns an (nxN) array 
        """
        counts = np.random.choice(np.arange(self.K), n, p = self.weights) 
        _, n_i = np.unique(counts, return_counts = True)

        x = [] #data 

        for i in range(self.K):
            x_i = np.random.multivariate_normal( self.means[i], self.cov[i], n_i[i])
            x += [x_i]
        x = np.vstack(x)
        return x
    
    def density(self,x):
        """
        Evaluate the mixture density at x
        """
        
        y = 0
        for i in range(self.K):
            y += self.weights[i]*stats.multivariate_normal.pdf(x, self.means[i],  self.cov[i])
        return y

        
    def grad(self,x):
        """
        Evaluate the gradient of the mixture density at x
        """
        y = np.zeros(x.shape)
        for i in range(self.K):
            y += self.weights[i]*self._component_grad(x, i)
        return y

    def _component_grad(self,x, i):
        """
        return the gradient of the ith component of the mixture   
        """
        return -np.matmul(self.inv_cov[i], x - self.means[i])*stats.multivariate_normal.pdf(x, self.means[i], self.cov[i])


    def hessian(self,x):
        """
        Evaluate the hessian of the mixture density at x
        """
        y = np.zeros((len(x), len(x)))
        for i in range(self.K):
            y += self.weights[i]*self._component_hessian(x,i)
        return y
        
    def _component_hessian(self,x,i):
        """
        return the hessian of the ith component of the mixture  
        """
        m1 = np.outer(x-self.means[i], x-self.means[i])
        m2 = np.matmul(self.inv_cov[i], m1) - np.eye(len(x)) 
        return stats.multivariate_normal.pdf(x, self.means[i], self.cov[i])*np.matmul(m2, self.inv_cov[i])

###### Functions for sampling data ###### 

def get_covariance_matrix(var1, var2, corr):
    """
    Get covariance matrix for X = [X_1, X_2] 
    var1: Var(X_1)
    var2: Var(X_2)
    corr: Corr(X_1, X_2)
    """
    covariance = corr*np.sqrt(var1)*np.sqrt(var2)
    return np.array([[var1, covariance], [covariance, var2]])


    
    
    