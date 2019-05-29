import numpy as np
from scipy import stats

class GaussianMixtureModel():
    """Density estimation with Gaussian Mixture Models (GMM).

    You can add new functions if you find it useful, but **do not** change
    the names or argument lists of the functions provided.
    """
    def __init__(self, X, K):
        """Initialise GMM class.

        Arguments:
          X -- data, N x D array
          K -- number of mixture components, int
        """
        self.X = X
        self.n = X.shape[0]
        self.D = X.shape[1]
        self.K = K
    def gaussian(self, mu, S):
            
        A = (2*np.pi)**self.D
        B = np.linalg.det(sigma) * A
        C = 1.0/np.sqrt(B)
        P = np.matrix(x-mu)
        gs = C * np.exp(-0.5* P *np.linalg.inv(sigma)* P.T)
        return gs
            
    def likelihood(self, mu, S, pi):
        rsp = 0.0
        for k in range(K):
            rsp += pi[k]*gaussian(self.X[i], mu[k], S[k])
        return rsp

    def E_step(self, mu, S, pi):
        """Compute the E step of the EM algorithm.

        Arguments:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array

        Returns:
          r_new -- updated component responsabilities, N x K array
        """
        # Assert that all arguments have the right shape
        assert(mu.shape == (self.K, self.D) and\
               S.shape  == (self.K, self.D, self.D) and\
               pi.shape == (self.K, 1))
        r_new = np.zeros((self.n, self.K))

        # Task 1: implement the E step and return updated responsabilities
        # Write your code from here...
        
        for i in range(self.n):
            for k in range(K):
                r_new[i][k] = pi[k]*self.gaussian(self.X[i], mu[k], S[k])/self.likelihood(self.X[i], K, mu, S, pi)
        
        #r_new = resp[i][k]
        # ... to here.
        assert(r_new.shape == (self.n, self.K))
        return r_new


    def M_step(self, mu, r):
        """Compute the M step of the EM algorithm.

        Arguments:
          mu -- previous component means, K x D array
          r -- previous component responsabilities,  N x K array

        Returns:
          mu_new -- updated component means, K x D array
          S_new -- updated component covariances, K x D x D array
          pi_new -- updated component weights, K x 1 array
        """
        assert(mu.shape == (self.K, self.D) and\
               r.shape  == (self.n, self.K))
        mu_new = np.zeros((self.K, self.D))
        S_new  = np.zeros((self.K, self.D, self.D))
        pi_new = np.zeros((self.K, 1))

        # Task 2: implement the M step and return updated mixture parameters
        # Write your code from here...
        for k in range(K):
            Nk = 0
            x_mu = np.zeros((1,self.D))
            for i in range(self.n):
                Nk += r[i][k]
                mu_new[k] += (r[i][k])*self.X[i]
            mu_new[k] /= Nk
            
            for i in range(self.n):
                x_mu[k] = self.X[i] - mu_new[k]
                S_new[k] += (r[i][k]/Nk)*np.dot(x_mu, x_mu.T)

            pi_new[k] = Nk/self.n

        # ... to here.
        assert(mu_new.shape == (self.K, self.D) and\
               S_new.shape  == (self.K, self.D, self.D) and\
               pi_new.shape == (self.K, 1))
        return mu_new, S_new, pi_new

    def log_likelihood(self, mu, S, pi):
            val = 0.0
            for i in range(self.n):
                somme = 0.0
                for k in range(K):
                    somme += pi[k] * gaussian(self.X[i], mu[k], S[k])
                val +=  log(somme)
            return - val
        
    def train(self, initial_params):
        """Fit a Gaussian Mixture Model (GMM) to the data in matrix X.

        Arguments:
          initial_params -- dictionary with fields 'mu', 'S', 'pi' and 'K'

        Returns:
          mu -- component means, K x D array
          S -- component covariances, K x D x D array
          pi -- component weights, K x 1 array
          r -- component responsabilities, N x K array
        """
        # Assert that initial_params has all the necessary fields
        assert(all([k in initial_params for k in ['mu', 'S', 'pi']]))

        mu = np.zeros((self.K, self.D))
        S  = np.zeros((self.K, self.D, self.D))
        pi = np.zeros((self.K, 1))
        r  = np.zeros((self.n, self.K))

        # Task 3: implement the EM loop to train the GMM
        # Write your code from here...
        mu, S, pi =self.M_step(mu, r)
        espcilon = 0.0001 
        mle_tab = []
        mle_tab.append(log_likelihood(mu, S, pi))
        
        r = self.E_step(mu, S, pi)
        mu, S, pi = self.M_step(mu, r)
        mle_tab.append(log_likelihood(mu, S, pi))
        i = 1
        while np.abs(mle_tab[i] - mle_tab[i-1]) > espcilon:
            r = self.E_step(mu, S, pi)
            mu, S, pi = self.M_step(mu, r)
            mle_tab.append(log_likelihood(mu, S, pi))
            i += 1 

        # ... to here.
        assert(mu.shape == (self.K, self.D) and\
               S.shape  == (self.K, self.D, self.D) and\
               pi.shape == (self.K, 1) and\
               r.shape  == (self.n, self.K))
        return mu, S, pi, r


if __name__ == '__main__':
    np.random.seed(43)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    # Do not write code outside the class definition or
    # this if-block.
    ##########################
