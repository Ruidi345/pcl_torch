import numpy as np
import torch

class Replay():
    def __init__(self, episode_info) -> None:
        
        if episode_info is None:
            self.empty = True
            self._initial_state = None
            self._observations = None
            self._actions = None
            # self._rewards = rewards
            self._rewards = None
            self._terminated = None
            self._pads = None
        else:
            self.empty = False
            (initial_state, observations, actions, rewards,
                terminated, pads) = episode_info
            # initial_state (list(Tensor(1, bs, internal_dim), Tensor(1, bs, internal_dim))):
            # observations (list(Tensor(bs), ... , Tensor(bs))): num of Tensor is time_steps+1
            # actions (list(Tensor(bs), ... , Tensor(bs))): num of Tensor is time_steps+1
            # rewards (ndarray(time_steps, bs)):
            # terminated (ndarray(bs)):
            # pads (ndarray(time_steps, bs)):
            self._initial_state = initial_state
            self._observations = observations
            self._actions = actions
            # self._rewards = rewards
            self._rewards = np.array(rewards)
            self._terminated = terminated
            self._pads = pads

            a = 1

    @property
    def batch_size(self):
        return self._rewards[0].shape[0]
    
    def initial_state(self):
        return self._initial_state
    
    def observations(self):
        return self._observations
    
    def actions(self):
        return self._actions

    def rewards(self):
        return self._rewards
    
    def terminated(self):
        return self._terminated
    
    def pads(self):
        return self._pads
    
    @property
    def max_legnth(self):
        return 0 
    
    @property
    def all_done(self):
        return 0 
    
    @staticmethod
    def convert_to_batched_episodes(episodes, max_length=None):
        """Convert batch-major list of episodes to time-major batch of episodes."""
        lengths = [len(ep[-2]) for ep in episodes]
        max_length = max_length or max(lengths)

        new_episodes = []
        for ep, length in zip(episodes, lengths):
            initial, observations, actions, rewards, terminated = ep
            observations = [np.resize(obs, [max_length + 1] + list(obs.shape)[1:])
                            for obs in observations]
            actions = [np.resize(act, [max_length + 1] + list(act.shape)[1:])
                       for act in actions]
            pads = np.array([0] * length + [1] * (max_length - length))
            rewards = np.resize(rewards, [max_length]) * (1 - pads)
            new_episodes.append([initial, observations, actions, rewards,
                                terminated, pads])
        # replay_initial:list, len=bs, elem: [tensor.shape=(1, 128), tensor.shape=(1, 128)]
        (repaly_initial, observations, actions, rewards,
         terminated, pads) = zip(*new_episodes)

        def trans(value):
            value = torch.split(value, 1, dim=0)
            value = [torch.squeeze(v) for v in value]
            return tuple(value)
        
        observations = [torch.transpose(torch.Tensor(obs), 0, 1) for obs in zip(*observations)]
        observations = [trans(obs) for obs in observations]

        actions = [torch.transpose(torch.Tensor(act), 0, 1) for act in zip(*actions)]
        actions = [trans(act) for act in actions]

        rewards = np.transpose(rewards)
        rewards = np.array_split(rewards, rewards.shape[0])
        rewards = [(*(np.squeeze(reward).tolist()),) for reward in rewards]

        pads = np.transpose(pads)
        pads = np.array_split(pads, pads.shape[0])
        pads = [np.squeeze(pad) for pad in pads]

        initial = [torch.transpose(torch.stack(ini), 1, 0) for ini in zip(*repaly_initial)]
          
        terminated = np.array(terminated)

        return (initial, observations, actions, rewards, terminated, pads)
    
    def to_episodes(self):
        """Convert time-major batch of episodes to batch-major list of episodes.
           convert_from_batched_episodes

        Args:
            initial_state (list(Tensor(1, bs, internal_dim), Tensor(1, bs, internal_dim))):
            observations (list(Tensor(bs), ... , Tensor(bs))): num of Tensor is time_steps+1
            actions (list(Tensor(bs), ... , Tensor(bs))): num of Tensor is time_steps+1
            rewards (ndarray(time_steps, bs)):
            terminated (ndarray(bs)):
            pads (ndarray(time_steps, bs)):
        """
        # TODO def of time major and batch major, 
        # TODO check the output of sampling episode
        rewards = np.array(self._rewards)
        pads = np.array(self._pads)
        num_episodes = rewards.shape[1]

        def convert(val):
            return np.array([a.numpy() for a in val])
        actions = [convert(act) for act in self._actions]
        observations = [convert(obs) for obs in self._observations]

        total_rewards = np.sum(rewards * (1 - pads), axis=0)

        total_length = np.sum(1 - pads, axis=0).astype('int32')
        episodes = []
        for i in range(num_episodes):

            length = total_length[i]

            ep_obs = [obs[:length + 1, i, ...] for obs in observations]
            ep_act = [act[:length + 1, i, ...] for act in actions]
            ep_initial = [state[:, i, :]
                          for state in self._initial_state]            
            ep_rewards = rewards[:length, i]

            episodes.append(
                [ep_initial, ep_obs, ep_act, ep_rewards, self._terminated[i]])

        return episodes