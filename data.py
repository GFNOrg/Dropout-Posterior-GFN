import numpy as np
from utils import enumerate_all_binary_matrices
from scipy.special import logsumexp


class GaussianBinaryMask:
    def __init__(self, d=3, n=2, A_std=5.):
        """
        :param d: dimension of z
        :param n: dimension of x = A z
        :param A_std: standard deviation s, such that A ~ N(0, s^2 I)
        """
        self.d = d
        self.n = n
        self.A = A_std * np.random.randn(n, d)
        while True:
            self.theta_true = np.random.randint(2, size=(n, d))
            if np.sum(self.theta_true) != 0:  # Making sure we are not dropping all connections
                break
        self.A_theta = self.A * self.theta_true
        self. X = None

    def sample(self, N=10):
        """
        :param N: number of iid samples
        :return: N x n numpy array representing the training data
        """
        Z = np.random.randn(N, self.d)
        X = Z @ self.A_theta.T
        self.X = X
        return X

    def loglikelihood(self, X, theta):
        """
        :param X: N' x n numpy array
        :param theta: n x d binary numpy array
        :return: log p(X | theta)
        """
        A_theta = self.A * theta
        const = np.linalg.det(A_theta @ A_theta.T)
        if const <= 0:
            return np.float('-inf')
        const *= (2 * np.pi) ** self.n
        const = 1. / np.sqrt(const)
        const = X.shape[0] * np.log(const)
        mat = X @ np.linalg.inv(A_theta @ A_theta.T) @ X.T
        return const - 0.5 * np.trace(mat)

    def evaluate_true_posterior(self, X, prior=None):
        """
        :param X: N' x n numpy array
        :param prior: numpy array representing the priors p(theta) for all 2 ^(n * d) values of theta. Uniform if None.
        :return: numpy array representing the posterior p(theta | X) for all 2^(n * d) values of theta
        """
        thetas = enumerate_all_binary_matrices(self.n, self.d)
        loglikelihoods = np.array([self.loglikelihood(X, theta) for theta in thetas])
        if prior is None:
            prior = 1. / (2 ** (self.n * self.d)) * np.ones(2 ** (self.n * self. d))
        logpriors = np.log(prior)
        numerator = loglikelihoods + logpriors
        logposterior = numerator - logsumexp(numerator)
        posterior = np.exp(logposterior)
        return posterior

