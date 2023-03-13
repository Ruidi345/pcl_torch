import gym 
import gym_algorithmic
import env_spec
import random
import time
import numpy as np
from six.moves import xrange

def get_env(env_str, base):
#   return gym.make(env_str, base=base)
  return gym.make(env_str)

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
            # action=(1,0,1)在下面的convert_action_to_gym转化成list

            action = self.env_spec.convert_action_to_gym(action)
            # self.check_nan(action)
            #! gym.step()最底层模拟函数
            obs, reward, done, tt = env.step(action)
            # 将obs整合成list
            obs = self.env_spec.convert_obs_to_list(obs)
            return obs, reward, done, tt

        # 将由list包裹的tf.tensor(shape=batch_size, act_dim)
        # ->
        # if not self.env_spec.try_combining_actions:
        #     actions = [act.numpy() for act in actions]

        actions = zip(*actions)
        # for action, env, done in zip(actions, self.envs, self.dones):
        # #   这里的action是一个tuple，len=env.action_dim
        #   if not done: # 场景还未结束sampling done=False
        #     env_step(env, action)
        #   else:
        #     (self.env_spec.initial_obs(None), 0, True, None)
        #     obs = (self.env_spec.initial_obs(None))
        #     reward = 0
        #     done = True
        #     info = None
        #! 这里的循环扫过batch中100个的每一个场景

        # self.dones作用于多个环境并行运行时，用于记录batch中每个单独环境是否已经结束，
        # ->因为可能存在某个环境还没结束，但其他的已经结束的情况，此时还需要继续运行step(),
        #   来继续模拟进行未结束的环境
        # ->而已结束的环境，统统使用人为设定的初始obs，reward为0，True结束作为返回值
        outputs = [env_step(env, action)
                if not done else (self.env_spec.initial_obs(None), 0, True, None)
                for action, env, done in zip(actions, self.envs, self.dones)]

        # output组成obs, reward, done, tt
        # 修改batch中每一个情景的结束标志
        # done = True: 1.来自环境，说明在这一步，模拟被终止
        #              2.由于self.dones为True，导致done是被上面手动控制为True
        for i, (_, _, done, _) in enumerate(outputs):
        # 只要有一个为True,情景停止，则self.dones[i]为True
            self.dones[i] = self.dones[i] or done

        # 这里是将obs=([2],[3],...)->[[2,3]]
        obs, reward, done, tt = zip(*outputs)
        obs = [list(oo) for oo in zip(*obs)]
        # obs = [list(oo)[0] for oo in zip(*obs)]

        #TODO tt?后面没用到，不用管
        # print(type(obs), len(obs))
        # print(type(obs[0]), len(obs[0]))
        # print(type(obs[0][0]), obs[0][0].shape)
        return [obs, reward, done, tt]

    def check_nan(self, action):
        res = np.isnan(action.numpy())
        if np.any(np.isnan(action.numpy())):
            print("nan value detected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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

        # 第一次self.count=1，不进入if分支
        #TODO self.count在那里控制
        if self.count != 1:
            assert np.all(predicate)
            return self.reset()

        # self.num_episodes_played = 100，这里是因为第一次predicate都为True
        self.num_episodes_played += sum(predicate)
        
        # self.envs是个list，其元素都是<TimeLimit<CopyEnv<Copy-v0>>>,len=100
        # for env, pred in zip(self.envs, predicate)
        #   if pred == Ture:
        #     self.env_spec.convert_obs_to_list(env.reset()
        #   else:
        #     None

        # self.env_spec是一个class，其包括self.env
        # self.env_spec = env_spec.EnvSpec(self.get_one())
        #TODO 可能需要查询Gym相关文档

        # env.reset() 返回observation结果，例如Copy环境的1，2，3，4各自代表环境中的状态
        # for env, pred in zip(self.envs, predicate):
        #     if pred == True:
        #         output = self.env_spec.convert_obs_to_list(env.reset())
        #     else:
        #         output = None

        #* 这里要求env.reset()要有合理反馈的obs，而在carla的初始化里可能会出问题，要检查一下
        output = [self.env_spec.convert_obs_to_list(env.reset())
                if pred else None
                for env, pred in zip(self.envs, predicate)]

        # self.dones:list类型， 元素为bool，len=100
        # 这里如果prediction[i]为True，则dones[i]取反
        for i, pred in enumerate(predicate):
            if pred:
                self.dones[i] = False
        # print(type(output),type(output[0]),type(output[0][0]))

        return output

