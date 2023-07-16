import torch
import torch.nn.functional as F
import numpy as np

class Objective():
    def __init__(self, learning_rate=0.02, clip_norm=5,
                policy_weight=1.0, critic_weight=1.0,
                tau=0.1, gamma=1.0, rollout=10,
                eps_lambda=0.0, clip_adv=None,
                use_target_values=False):
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout
        self.loss_func = None
        self.kLossCoeff = 2 / 2 ** 0.5

    def get_loss(self):
        raise NotImplementedError()
    
    def get_expected_from_values(self):
        raise NotImplementedError()

    def get_targets_from_reward_and_logprob(self):
        raise NotImplementedError()

    def discounted_future_sum(self, values, discount, rollout):
        """Discounted future sum of time-major values."""
        discount_filter = torch.reshape(
            discount ** torch.arange(float(rollout)), [1, 1, -1])
        expanded_values = torch.cat(
            [values, torch.zeros([rollout - 1, values.shape[1]])], 0)

        transposed_expand_values = expanded_values.transpose(0, 1)
        conv1d_input = torch.unsqueeze(transposed_expand_values, -1)
        transposed_conv1d_input = conv1d_input.transpose(1, 2)
        squeeze_input = torch.nn.functional.conv1d(input=transposed_conv1d_input, weight=discount_filter, stride=1, padding='valid')
        conv_values = torch.squeeze(squeeze_input, 1).transpose(0, 1)

        return conv_values

    def shift_values(self, values, gamma, rollout, final_values=0.0):
        """Shift values up by some amount of time.

        Those values that shift from a value beyond the last value
        are calculated using final_values.

        """
        roll_range = torch.cumsum(torch.ones_like(values[:rollout, :]), axis=0)

        final_pad = torch.unsqueeze(final_values, 0) * gamma ** roll_range
        return torch.cat([gamma ** rollout * values[rollout:, :], final_pad], 0)
    
class PCL(Objective):
    def __init__(self, learning_rate=0.02, clip_norm=5,
                policy_weight=1.0, critic_weight=1.0,
                tau=0.1, gamma=1.0, rollout=10,
                eps_lambda=0.0, clip_adv=None,
                use_target_values=False):
        super(PCL, self).__init__()

        self.loss_func = F.mse_loss

    def cut_values(self, values, terminated):
        terminated = torch.Tensor(terminated)
        return values[:-1, :], values[-1, :] * (1 - terminated)

    def get_loss(self, values, replay, log_probs):
        values_, final_values = self.cut_values(values, replay.terminated())
        return self.loss_func(self.kLossCoeff * self.get_expected_from_values(values_, final_values, replay.pads()), 
                              self.kLossCoeff * self.get_targets_from_reward_and_logprob(replay.rewards(), log_probs, replay.pads()))
    
    def get_expected_from_values(self, values, final_values, pads):
        pads = torch.Tensor(np.array(pads))
        pads = pads.to(torch.float32)
        
        not_pad = 1 - pads
        batch_size = pads.shape[1]  
        value_estimates    = not_pad * values

        value_estimates = torch.cat(
            [self.gamma ** torch.unsqueeze(
                torch.arange(float(self.rollout - 1), 0, -1), dim=1) *
            torch.ones([self.rollout - 1, batch_size]) *
            value_estimates[0:1, :],
            value_estimates], dim=0)

        roll_range_ = torch.cumsum(torch.ones_like(value_estimates[:self.rollout, :]), axis=0)
        final_pad_ = torch.unsqueeze(final_values, 0) * self.gamma ** roll_range_
        last_values = torch.cat([self.gamma ** self.rollout * value_estimates[self.rollout:, :],
                            final_pad_], 0)
        
        baseline_values = value_estimates - last_values

        return baseline_values

    def get_targets_from_reward_and_logprob(self, rewards, log_probs, pads):
        pads = torch.Tensor(np.array(pads))
        pads = pads.to(torch.float32)
        rewards = torch.Tensor(np.array(rewards))
        rewards = rewards.to(torch.float32)
        
        not_pad = 1 - pads
        batch_size = pads.shape[1] 
        rewards            = not_pad * rewards
        log_probs = [torch.stack(v) for v in zip(*log_probs)]
        log_probs = not_pad * sum(log_probs)

        not_pad = torch.cat([torch.ones([self.rollout - 1, batch_size]),
                            not_pad], 0)
        rewards = torch.cat([torch.zeros([self.rollout - 1, batch_size]),
                            rewards], 0)

        log_probs = torch.cat([torch.zeros([self.rollout - 1, batch_size]),
                            log_probs], 0)

        sum_rewards            = self.discounted_future_sum(rewards, self.gamma, self.rollout)
        sum_log_probs          = self.discounted_future_sum(log_probs, self.gamma, self.rollout)
        
        future_values = (
            - self.tau * sum_log_probs
            + sum_rewards)

        return future_values