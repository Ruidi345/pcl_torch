import gym 
import gym_algorithmic
import env_spec
import random
import time
import numpy as np
from six.moves import xrange

import torch
def get_env(env_str, base):
#   return gym.make(env_str, base=base)
  return gym.make(env_str)

def tensor_to_tuple(t_in_list):
    l = torch.Tensor(t_in_list)
    # TODO gpu detach to cpu and detach  
    l = l.numpy().astype(int)

    return tuple(l)

class Environment():
    def __init__(self, env_name, base=4 ,distinct=400, count=1, seeds=None):


        self.distinct = distinct # should be batch_size // self.num_samples
        self.count = count
        self.total = self.distinct * self.count
        self.seeds = seeds or [random.randint(0, 1e12)
                            for _ in xrange(self.distinct)]
        self.env_name = env_name

        self.envs = []
        for seed in self.seeds:
            for _ in xrange(self.count):
                env = get_env(self.env_name, base)
                env.seed(seed)
                if hasattr(env, 'last'):
                    env.last = 100  # for algorithmic envs
                self.envs.append(env)

        self.dones = [True] * self.total
        self.num_episodes_played = 0

        one_env = self.get_one()
        self.use_action_list = hasattr(one_env.action_space, 'spaces')
        self.env_spec = env_spec.EnvSpec(self.get_one())

    def step(self, actions):
        """action->env->state,re,done,_ for all env in one batch"""

        def env_step(env, action):

            action = self.env_spec.convert_action_to_gym(action)

            action = tensor_to_tuple(action)
            obs, reward, done, tt = env.step(action)
            obs = self.env_spec.convert_obs_to_list(obs)
            return obs, reward, done, tt


        actions = zip(*actions)
        outputs = [env_step(env, action)
                if not done else (self.env_spec.initial_obs(None), 0, True, None)
                for action, env, done in zip(actions, self.envs, self.dones)]

        for i, (_, _, done, _) in enumerate(outputs):
            self.dones[i] = self.dones[i] or done

        obs, reward, done, tt = zip(*outputs)
        obs = [list(oo) for oo in zip(*obs)]

        return [obs, reward, done, tt]

    def check_nan(self, action):
        res = np.isnan(action.numpy())
        if np.any(np.isnan(action.numpy())):
            print(action.numpy())


    def get_one(self):
        return random.choice(self.envs)
    
    def __len__(self):
        return len(self.envs)

    def all_done(self):
        return all(self.dones)

    # 没有引用
    def get_seeds(self):
        return self.seeds

    def reset_if(self, predicate=None):
        if predicate is None:
            predicate = self.dones

        if self.count != 1:
            assert np.all(predicate)
            return self.reset()

        self.num_episodes_played += sum(predicate)
        
        output = [self.env_spec.convert_obs_to_list(env.reset())
                if pred else None
                for env, pred in zip(self.envs, predicate)]

        for i, pred in enumerate(predicate):
            if pred:
                self.dones[i] = False

        return output

