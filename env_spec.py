# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for environment interface with agent / tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym
import gym_algorithmic
from six.moves import xrange


class spaces(object):
  discrete = 0
  box = 1


def get_space(space):
  # 离散进入这里
  if hasattr(space, 'n'):
    return space.n, spaces.discrete, None
  # 连续进入这里
  elif hasattr(space, 'shape'):
    return np.prod(space.shape), spaces.box, (space.low, space.high)


def get_spaces(spaces):
  # 多个离散的act组合进入这里，例如Copy
  if hasattr(spaces, 'spaces'):
    return zip(*[get_space(space) for space in spaces.spaces])
  else:
    # 连续和离散的环境都进入这里
    # 离散的单个obs空间也进入这里
    return [(ret,) for ret in get_space(spaces)]


class EnvSpec(object):

  def __init__(self, env, try_combining_actions=True,
               discretize_actions=None):
    self.discretize_actions = discretize_actions

    # try_combining_actions = False
    # self.try_combining_actions = try_combining_actions
    # figure out observation space
    self.obs_space = env.observation_space
    self.obs_dims, self.obs_types, self.obs_info = get_spaces(self.obs_space)
    # self.obs_dims = (4,)
    # self.obs_types = (1,)
    # self.obs_info = (array([-inf, -inf, -inf, -inf]), array([inf, inf, inf, inf]))

    # figure out action space
    self.act_space = env.action_space
    self.act_dims, self.act_types, self.act_info = get_spaces(self.act_space)
    # self.obs_dims = (1,)
    # self.obs_types = (1,)
    # self.obs_info = (array([-3]), array([3]))

    if self.discretize_actions:
      self._act_dims = self.act_dims[:]
      self._act_types = self.act_types[:]
      self.act_dims = []
      self.act_types = []
      for i, (dim, typ) in enumerate(zip(self._act_dims, self._act_types)):
        if typ == spaces.discrete:
          self.act_dims.append(dim)
          self.act_types.append(spaces.discrete)
        elif typ == spaces.box:
          for _ in xrange(dim):
            self.act_dims.append(self.discretize_actions)
            self.act_types.append(spaces.discrete)
    else:
      # inverseP进入这里
      self._act_dims = None
      self._act_types = None

    # 所有action都是离散时进入这里
    if (try_combining_actions and
        all(typ == spaces.discrete for typ in self.act_types)):
      self.combine_actions = True
      self.orig_act_dims = self.act_dims[:]
      self.orig_act_types = self.act_types[:]
      total_act_dim = 1
      for dim in self.act_dims:
        total_act_dim *= dim
      self.act_dims = [total_act_dim]
      self.act_types = [spaces.discrete]
    else:
      # inverseP进入这里
      self.combine_actions = False

    self.obs_dims_and_types = tuple(zip(self.obs_dims, self.obs_types))
    self.act_dims_and_types = tuple(zip(self.act_dims, self.act_types))

    # 这个attr没用
    self.total_obs_dim = sum(self.obs_dims)

    # 1.PCLModel中的output_dim
    # 2.PCLModel中的process_net_outputs有assert
    self.total_sampling_act_dim = sum(self.sampling_dim(dim, typ)
                                      for dim, typ in self.act_dims_and_types)

    # 这个attr没用
    self.total_sampled_act_dim = sum(self.act_dims)

    # print('discretize_actions', self.discretize_actions)
    # print('obs_space', self.obs_space)
    # print('obs_dims', self.obs_dims) #(76,) | ((6, 0),)
    # print('obs_types', self.obs_types) #(1,) | ((6, 0),)
    # print('obs_info', self.obs_info) 
    # print('act_space', self.act_space) 
    # print('act_dims', self.act_dims) # (2,) | [20]
    # print('act_types', self.act_types) # (1,) | [0]
    # print('act_info', self.act_info)
    # print('combine_actions', self.combine_actions) # False | True
    # # print('orig_act_dims', self.orig_act_dims) # (2, 2, 5)
    # # print('orig_act_types', self.orig_act_types) # (0, 0, 0)
    # print('obs_dims_and_types', self.obs_dims_and_types) #((8, 1),) | ((6, 0),)
    # print('act_dims_and_types', self.act_dims_and_types) #((2, 1),) | ((20, 0),)
    # print('total_obs_dim', self.total_obs_dim) # 8 | 6
    # print('total_sampling_act_dim', self.total_sampling_act_dim) # 4 | 20
    a = 1


  # 调用: 1. model.process_net_outputs
  #       离散就等于实际dim， 连续时候为两倍，分别是mean和std_dev
  #       之后分别在logit， entropy等的计算中使用
  def sampling_dim(self, dim, typ):
    if typ == spaces.discrete:
      return dim
    elif typ == spaces.box:
      return 2 * dim  # Gaussian mean and std
    else:
      assert False

  #!
  # 调用: 1. controller._sampling_episode
  def convert_actions_to_env(self, actions):
    # 在inverseP中直接跳过下面的两个分支
    #* 但是这里可能在离散中进入，还需继续查看
    if self.combine_actions:
      new_actions = []
      # 将action外层包裹的多余一层tuple去掉
      actions = actions[0]
      # Copy env下:self.orig_act_dims=(2,2,5)
      #* decode神经网络的输出，通过取余数，将标量转换成符合action dim的向量
      # decoding后，action(100,1)->(100,3(action_dim))
      for dim in self.orig_act_dims:
        new_actions.append(np.mod(actions, dim))
        actions = tf.cast(actions / dim, tf.int32)
      actions = new_actions

    # 下面的if分支被跳过，self.discretize_actions在构建时，默认为None
    if self.discretize_actions:
      new_actions = []
      idx = 0
      for i, (dim, typ) in enumerate(zip(self._act_dims, self._act_types)):
        if typ == spaces.discrete:
          new_actions.append(actions[idx])
          idx += 1
        elif typ == spaces.box:
          low, high = self.act_info[i]
          cur_action = []
          for j in xrange(dim):
            cur_action.append(
                low[j] + (high[j] - low[j]) * actions[idx] /
                float(self.discretize_actions))
            idx += 1
          new_actions.append(np.hstack(cur_action))
      actions = new_actions

    return actions

  # 没有调用
  def convert_env_actions_to_actions(self, actions):

    if not self.combine_actions:
      return actions

    new_actions = 0
    base = 1
    for act, dim in zip(actions, self.orig_act_dims):
      new_actions = new_actions + base * act
      base *= dim

    return [new_actions]

  # 调用 在GymWrapper里面有用， 其余没有
  def convert_obs_to_list(self, obs):
    if len(self.obs_dims) == 1:
      return [obs]
    else:
      return list(obs)

  # 调用 在GymWrapper里面有用， 其余没有
  def convert_action_to_gym(self, action):
    if len(action) == 1:
      return action[0]
    else:
      return list(action)
    if ((not self.combine_actions or len(self.orig_act_dims) == 1) and
        (len(self.act_dims) == 1 or
         (self.discretize_actions and len(self._act_dims) == 1))):
      return action[0]
    else:
      return list(action)

  # 调用 1.在GymWrapper里面有用，
  #     2.Controller的init里面有用
  def initial_obs(self, batch_size):
    if batch_size == None:
      a = 1
    batched = batch_size is not None
    batch_size = batch_size or 1

    obs = []
    for dim, typ in self.obs_dims_and_types:
      if typ == spaces.discrete:
        obs.append(np.zeros(batch_size,dtype=np.int32))
      elif typ == spaces.box:
        obs.append(np.zeros([batch_size, dim],dtype=np.int32))

    if batched:
      return obs
    else:
      return list(zip(*obs))[0]

  # 调用 1.Controller的init里面有用
  def initial_act(self, batch_size=None):
    batched = batch_size is not None
    batch_size = batch_size or 1

    act = []
    for dim, typ in self.act_dims_and_types:
      if typ == spaces.discrete:
        act.append(-np.zeros(batch_size,dtype=np.int64))
      elif typ == spaces.box:
        act.append(-np.zeros([batch_size, dim],dtype=np.float32))

    if batched:
      return act
    else:
      return list(zip(*act))[0]

  def is_discrete(self, typ):
    return typ == spaces.discrete

  def is_box(self, typ):
    return typ == spaces.box


if __name__ == '__main__':
  # env = gym.make('Copy-v0', base=5)
  # env = gym.make('Copy-v0', base=7)
  env = gym.make('RepeatCopy-v0')
  enc_spe = EnvSpec(env)
