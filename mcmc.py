import numpy as np


class DiscreteMetropolisHastings:
    def __init__(self, K, loglikelihood_fn, prior=None, proposal=None):
        """
        :param K: cardinality of the set the distribution is defined on
        :param loglikelihood_fn: function mapping an index to a loglikelihood log p(data|index)
        :param prior: prior distribution P(index). Uniform if None
        :param proposal: proposal distribution Q(index_new | index_current) represented as K x K array. Uniform if None
        """
        self.K = K
        self.likelihood_fn = loglikelihood_fn
        if prior is None:
            prior = 1. / K * np.ones(K)
        self.prior = prior
        if proposal is None:
            proposal = 1. / K * np.ones((K, K))
        self.proposal = proposal

    def sample_from_proposal(self, current_index):
        """
        :param current_index: integer in {0, ..., K - 1}
        :return: proposed index
        """
        probs = self.proposal[current_index]
        proposed_index = np.random.choice(np.arange(self.K), p=probs)
        return proposed_index

    def acceptance(self, current_index, proposed_index):
        """
        :param current_index: current index
        :param proposed_index: proposed index
        :return: True if the MH condition is satisfied
        """
        current_loglikelihood = self.likelihood_fn(current_index)
        proposed_loglikelihood = self.likelihood_fn(proposed_index)
        logratio = (proposed_loglikelihood + np.log(self.prior[proposed_index]) -
                    (current_loglikelihood + np.log(self.prior[current_index])))
        if logratio > 0:
            return True
        else:
            return np.random.uniform(0, 1) < np.exp(logratio)

    def iterate(self, index_init, num_iterations):
        """
        :param index_init: initial index (usually random)
        :param num_iterations: number of MH steps
        :return: 2 lists: list of accepted and rejected sampled indices
        """
        current_index = index_init
        accepted = []
        rejected = []
        for i in range(num_iterations):
            proposed_index = self.sample_from_proposal(current_index)
            accept = self.acceptance(current_index, proposed_index)
            if accept:
                accepted.append(proposed_index)
                current_index = proposed_index
            else:
                rejected.append(proposed_index)

        return accepted, rejected

    def approximate_posterior(self, index_init, num_iterations):
        """
        :param index_init: same as for self.iterate
        :param num_iterations: same as for self.iterate
        :return: numpy array of size K representing approximated posterior probabilities p(index | data)
        """
        accepted, _ = self.iterate(index_init, num_iterations)
        induced_posterior = [accepted.count(i) / len(accepted) for i in range(self.K)]

        return induced_posterior
