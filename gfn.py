import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


class GFN(nn.Module):
    def __init__(self, m, learn_Z=False, architecture=None, h=None):
        """
        :param architecture: nn.Module with input layer of size m and output layer of size m+1
        :param m: input size
        :param learn_Z: if True, then an extra parameter needs to be learned logZ
        :param h: if architecture is not specified then it's the number of hidden units for each of the 2 hidden layers
        """
        super().__init__()
        self.m = m
        if architecture is None:
            assert h is not None
            architecture = nn.Sequential(nn.Linear(m, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, m + 1))
        self.logit_maker = architecture
        if learn_Z:
            self.logZ = nn.Parameter(torch.tensor(0.))

    def forward(self, s):
        """
        :param s: (batched) states of the GFN. tensor of size k x m
        :return: P_F(. | s) represented as a tensor of size (k + 1) x m. Invalid action values are replaced with -inf????
        """
        assert s.ndim == 2 and s.shape[1] == self.m
        logits = self.logit_maker(s)
        return logits


class BinaryMaskGFN(GFN):
    """
    The states of the GFN are binary vectors of size m. An action amounts to replacing a zero with a one
    """

    def forward(self, theta):
        """
        :param theta: (batched) states of the GFN, binary tensor of size k x m
        :return: masked out logits: -infinity to invalid actions (those that try to replace a 1 with a 1)
        """
        logits = super().forward(theta)
        logits[torch.cat((theta, torch.zeros(theta.shape[0], 1)), 1) == 1.] = - float("inf")
        return logits

    def on_policy_trajectory_sample(self, temperature=1.):
        """
        :param temperature: softmax temperature
        :return: a complete GFN trajectory (i.e. ending in a terminating state) as a T x m tensor
        The function assumes the source state of the GFN is [0, 0, ..., 0]
        """

        theta = torch.zeros(1, self.m)
        trajectory = [theta.clone().detach()]
        while True:
            logits = self(theta)
            probs = torch.softmax(logits / temperature, 1)
            idx = Categorical(probs).sample()[0]
            if idx == self.m:  # terminating state
                break
            theta[0, idx] = 1.
            trajectory.append(theta.clone().detach())
        return torch.cat(trajectory, 0)

    def uniform_trajectory_sample(self, terminate_prob=0.1):
        """
        :param terminate_prob: probability of ending the trajectory at each step of the trajectory
        :return: a complete GFN trajectory (i.e. ending in a terminating state) as a T x m tensor
        """
        m = self.m
        theta = torch.zeros(1, m)
        trajectory = [theta.clone().detach()]
        while True:
            possible_indices = [i for i, j in enumerate(theta[0].detach().numpy()) if j == 0]
            if len(possible_indices) == 0:
                break
            probs = ((1 - terminate_prob) / len(possible_indices)) * np.ones(len(possible_indices))
            idx = np.random.choice(possible_indices + [m], p=list(probs) + [terminate_prob])
            if idx == m:
                break
            theta[0, idx] = 1.
            trajectory.append(theta.clone().detach())
        return torch.cat(trajectory, 0)


def modified_trajectory_balance_loss(pf, traj, MIN_REW, pb_fn, reward_fn):
    """
    :param pf: nn.Module representing P_F (can be an instantiation of BinaryMaskGFN e.g.)
    :param traj: sampled trajectory
    :param MIN_REW: minimum reward value to replace lower rewards (including -infinity) with
    :param pb_fn: function taking a trajectory (T x m tensor) as input and outputting the backward probs as a (T-1) tensor
    :param reward_fn: function mapping a trajectory to a reward (list of same length)
    :return: loss(traj)
    Works only if all states are connected to the sink state, those that aren't are given an imaginary reward of MIN_REW
    """
    trajectory_logits = pf(traj)
    final_forward_logprob_differences = trajectory_logits[:-1, -1] - trajectory_logits[1:, -1]
    denominators = torch.logsumexp(trajectory_logits, 1)
    final_forward_logprob_differences = final_forward_logprob_differences - (denominators[:-1] - denominators[1:])
    forward_logprobs = trajectory_logits[:-1, :-1][
        torch.arange(traj.shape[0] - 1), (traj[1:, :] - traj[:-1, :]).argmax(1)]  # 1-dim tensor of size T-1
    forward_logprobs = forward_logprobs - denominators[:-1]

    pb = pb_fn(traj)
    log_pb = torch.log(pb)
    pred = final_forward_logprob_differences + log_pb - forward_logprobs

    rewards = reward_fn(traj)
    trajectory_rewards = torch.tensor(rewards).clamp_min(MIN_REW).to(pred.dtype)

    targets = trajectory_rewards[:-1] - trajectory_rewards[1:]
    loss = nn.MSELoss(reduction='sum')(pred, targets)

    return loss, rewards


def trajectory_balance_loss(pf, traj, pb_fn, reward_fn):
    """
    :param pf: nn.Module representing P_F (can be an instantiation of BinaryMaskGFN e.g.)
    :param traj: sampled trajectory
    :param MIN_REW: minimum reward value to replace lower rewards (including -infinity) with
    :param pb_fn: function taking a trajectory (T x m tensor) as input and outputting the backward probs as a (T-1) tensor
    :param reward_fn: function mapping a trajectory to a reward (list of same length)
    :return: loss(traj)
    needs the GFN to be paramertrized with logZ
    """
    trajectory_logits = pf(traj)
    denominators = torch.logsumexp(trajectory_logits, 1)
    forward_logprobs = trajectory_logits[:-1, :-1][
        torch.arange(traj.shape[0] - 1), (traj[1:, :] - traj[:-1, :]).argmax(1)]  # 1-dim tensor of size length-1
    forward_logprobs = forward_logprobs - denominators[:-1]

    sink_logprob = trajectory_logits[-1, -1] - denominators[-1]

    pb = pb_fn(traj)
    log_pb = torch.log(pb)

    pred = pf.logZ + forward_logprobs.sum() + sink_logprob - log_pb.sum()

    trajectory_rewards = reward_fn(traj)
    targets = torch.tensor(trajectory_rewards).to(pred.dtype)

    loss = nn.MSELoss()(pred, targets)

    return loss
