import numpy as np


def make_binary_matrix_from_binary_string(binary_str, d):
    """
    :param binary_str: string of length (n * d)
    :param d: number of columns of the wanted matrix
    :return: numpy array of size n x d
    """
    parts = [' '.join(binary_str[i:i + d]) for i in range(0, len(binary_str), d)]
    return np.array(np.matrix('; '.join(parts)))


def enumerate_all_binary_matrices(n, d):
    """
    :param n: number of rows
    :param d: number of columns
    :return: list of 2^(n * d) arrays of size n x d representing all possible binary matrices of size n x d
    """
    m = n * d
    thetas = []
    for i in range(2 ** m):
        binary = np.binary_repr(i).zfill(m)
        thetas.append(make_binary_matrix_from_binary_string(binary, d))
    return thetas


def index_to_binary_mask_loglikelihood_fn(data_maker, X=None):
    """
    :param X: dataset with which to evaluate loglikelihoods. If None, all available data is used
    :param data_maker: instance of GaussianBinaryMask
    :return: function mapping indices to corresponding loglikelihoods
    """
    if X is None:
        assert data_maker.X is not None, "Either specify X, or generate data with data maker first"
        X = data_maker.X

    def loglikelihood_fn(index):
        binary = np.binary_repr(index).zfill(data_maker.n * data_maker.d)
        theta = make_binary_matrix_from_binary_string(binary, data_maker.d)
        return data_maker.loglikelihood(X, theta)

    return loglikelihood_fn


def uniform_backwards_prob(traj):
    """
    :param traj: tensor of size T x m representing a trajectory with binary thetas
    :return: In the GaussianBinaryMask setting, returns a uniform backward probability amongst possible binary thetas
    """
    return 1. / traj[1:, :].sum(1)


def trajectory_loglikelihood_reward_fn(data_maker, X=None, prior=None):
    """
    :param X: dataset with which to evaluate loglikelihoods. If None, all available data is used
    :param data_maker: instance of GaussianBinaryMask
    :param prior: function mapping binary thetas to logpriors. If None, uniform on 2 ** m
    :return: In the GaussianBinaryMask setting, returns a function mapping trajectories to loglikelihoods + logpriors
    """
    if X is None:
        assert data_maker.X is not None, "Either specify X, or generate data with data maker first"
        X = data_maker.X

    n = data_maker.n
    d = data_maker.d

    if prior is None:
        def prior(_):
            return np.log(1. / 2 ** (n * d))

    def reward_fn(traj):
        length = traj.shape[0]
        traj_loglikelihoods = [data_maker.loglikelihood(X, traj[i].detach().numpy().reshape(n, d)) + prior(traj[i])
                               for i in range(length)]
        return traj_loglikelihoods

    return reward_fn
