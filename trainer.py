import argparse
import datetime
import os
from distutils.command.build import build

import numpy as np
import yaml

import controller_torch
import env_spec
import expert_paths
import gym_wrapper
import replay_buffer

current_time = datetime.datetime.now().strftime("%m%d-%H%M")

class Trainer():
    def __init__(self):
        # # args for Controller
        self.batch_size = 400
        self.env_name = 'Copy-v0'
        self.validation_frequency = 50
        self.base = 4

        self.hparams = dict((attr, getattr(self, attr))
                            for attr in dir(self)
                            if not attr.startswith('__') and
                            not callable(getattr(self, attr)))
        if 'summary_writer' in dir(self):
            self.hparams.pop('summary_writer')


        self.env = self.get_env()
        self.env_spec = env_spec.EnvSpec(self.env.get_one())

        self.crtl = controller_torch.Controller(env=self.env, batch_size=self.batch_size)
        self.num_steps = 10

    def get_env(self):
        return gym_wrapper.Environment(self.env_name, self.base, self.batch_size)


    def get_buffer_seeds(self):
        return expert_paths.sample_expert_paths(
            self.num_expert_paths, self.env_name, self.env_spec)

    def run(self):
        losses = []
        rewards = []
        output_reward = []
        all_ep_rewards = []
        # if self.train:
        #     self.crtl.set_writer(self.summary_writer)
        info_template = 'step {:d}, loss:{:.4f}, rewards:{:.4f}, ep rewards:{:.4f}, last ep reward: {:.4f}'
        print('hparams:\n%s', self.hparams_string())

        for cur_step in range(self.num_steps):
            loss ,total_rewards, episode_rewards= self.crtl.train(cur_step)

            losses.append(loss)
            rewards.append(total_rewards)

            if cur_step % self.validation_frequency == 0:
                all_ep_rewards.extend(episode_rewards)
                # print(info_template.format\
                #             (cur_step, 
                #             float(np.mean(losses)), 
                #             float(np.mean(rewards)),
                #             float(np.mean(all_ep_rewards)) if len(all_ep_rewards)>0 else 0,
                #             all_ep_rewards[-1] if len(all_ep_rewards) > 0 else 0
                #             ))

                output_reward.append(np.mean(rewards))                
                losses = []
                rewards = []
                all_ep_rewards = []

    def hparams_string(self):
        print(self.hparams)
        return '\n'.join('%s: %s' % item for item in sorted(self.hparams.items()))

class CFG(object):
    def __init__(self, cfgdict):
        self.__dict__.update(cfgdict)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
