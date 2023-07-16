import numpy as np
import torch
import torch.nn.functional as F
import env_spec

import replay_buffer

from torch_layers import Dense, LSTM, NewPolicyEncoder, spmax_tau
from torch_critic import NewCritic
from torch_policy import NewPolicy
from loss_objective import PCL
from replay import Replay


class Controller():
    def __init__(self,batch_size=400,
            env = None,
            unify_episodes = False,
            max_step = 200,
            replay_batch_size = -1,
            prioritize_by = 'rewards',
            validation_frequency = 250,
            get_buffer_seeds=None,):
        
        self.env = env
        self.env_spec = env_spec.EnvSpec(self.env.get_one())

        self.FLAG_batch_by_steps = False

        self.start_id = 0
        self.cutoff_agent = 0

        self.prioritize_by = prioritize_by
        self.FLAG_unify_episodes = unify_episodes
   
        # take this to buffer init  
        self.replay_buffer_size = 400
        self.replay_buffer_alpha = 0.5
        self.eviction = 'rand'

        self.loss_func = PCL()

        # self.num_steps = 10
        self.max_step = max_step

        self.batch_size = batch_size
        self.replay_batch_size = self.batch_size if replay_batch_size<0 else replay_batch_size

        self.replay_buffer = replay_buffer.PrioritizedReplayBuffer(max_size = self.replay_buffer_size, alpha = self.replay_buffer_alpha,
                                eviction_strategy = self.eviction) 

        hidden_size = 128
        internal_dim = 128
        self.policy = NewPolicy(env_spec=env_spec.EnvSpec(env.get_one()), 
                                internal_dim=internal_dim, 
                                input_prev_actions=True,
                                lstm_hidden_size=hidden_size,
                                test=True)
        self.critic = NewCritic(env_spec=env_spec.EnvSpec(env.get_one()), policy_cell_num=internal_dim)


        self.validation_frequency = validation_frequency
        if get_buffer_seeds is not None:
            self.seed_replay_buffer(get_buffer_seeds())

        self.learning_rate = 1e-5
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.parameters()},
                        {'params': self.critic.parameters()}
                    ], lr=self.learning_rate)
        
        self.init_episode_sampling_values(len(self.env))
        
    def init_episode_sampling_values(self, batch_size):
        # dont need this for simple PCL
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_running_rewards = np.zeros(batch_size)
        self.episode_running_lengths = np.zeros(batch_size)
        self.step_count = np.array([0] * batch_size)
        self.start_episode = np.array([True] * batch_size)
        # take these init values in to func or env_spec
        self.internal_state = np.array([self.initial_internal_state()] *
                                       batch_size)
        self.last_obs = self.env_spec.initial_obs(batch_size)
        self.last_act = self.env_spec.initial_act(batch_size)
        self.last_pad = np.zeros(batch_size)

    def sampling_episode(self, ):
        total_steps = 0
        while total_steps < self.max_step * len(self.env):
            # sampling for all batches
            initial_state, observations, actions, rewards, pads = self._sampling_episode()

            observations = list(zip(*observations))
            actions = list(zip(*actions))
            # self.env.dones 一个长度为bs的array，bool类型
            terminated = np.array(self.env.dones)

            self.get_episode_reward_lengths(rewards, pads, terminated)

            total_steps += np.sum(1 - np.array(pads))

            if not self.FLAG_batch_by_steps:
                return (initial_state,
                        observations, actions, rewards, terminated, pads)                     

    def _sampling_episode(self):
        all_obs, all_act, all_pad, rewards, = [], [], [], [], 
        all_log_probs = []
        step = 0

        obs_after_reset = self.env.reset_if(self.start_episode)

        for i, obs in enumerate(obs_after_reset):
            # i: batch size
            if obs is not None:
                self.step_count[i] = 0
                self.internal_state[i] = self.initial_internal_state()
                for j in range(len(self.env_spec.obs_dims)):
                    # self.last_obs.shape=(obs_dim, batch_size)
                    self.last_obs[j][i] = obs[j]
                for _ in range(len(self.env_spec.act_dims)):
                    # self.last_act[j][i] = -1
                    self.last_pad[i] = 0


        initial_state_h = torch.Tensor(self.internal_state[:, 0:1])
        initial_state_c = torch.Tensor(self.internal_state[:, 1:])
        initial_state_h = torch.transpose(initial_state_h, 0, 1)
        initial_state_c = torch.transpose(initial_state_c, 0, 1)
        self.internal_state = [initial_state_h, initial_state_c]
        initial_state = self.internal_state

        all_act.append(self.last_act)

        # 每循环一次，推进的是episode中的一个时间步
        while not self.env.all_done():
            self.step_count += 1 - np.array(self.env.dones)

            [state_h , state_c] = self.internal_state

            next_internal_state, sampled_actions, log_probs = \
                self.policy.sample_step(self.last_obs, self.last_act, (state_h, state_c))
                                        
            all_act.append(sampled_actions)

            # take action in env and get states
            next_obs, reward, next_dones, _ = self.env.step(
                self.env_spec.convert_actions_to_env(sampled_actions))
            
            # TODO also the reward and next_dones inton tensor?
            next_obs = [torch.Tensor(obs) for obs in next_obs]
            all_obs.append(self.last_obs)
            all_pad.append(self.last_pad)
            rewards.append(reward)

            all_log_probs.append(log_probs)

            self.internal_state = next_internal_state
            self.last_obs = next_obs
            self.last_act = sampled_actions
            self.last_pad = np.array(next_dones).astype(
                'float32')

            step += 1
            if self.max_step and step >= self.max_step:
                break
            
            # append final observation
        all_obs.append(self.last_obs)

        # reset for nect round sampling
        # TODO may warp this into a func
        self.internal_state = np.array(
            [self.initial_internal_state()] * len(self.env))

        return initial_state, all_obs, all_act, rewards, all_pad

    def get_log_probs_and_states_from_policy(self, replay):
        all_log_probs, all_states = [], []
        cur_state = replay.initial_state()
        all_states.append(replay.initial_state())

        observations = replay.observations()
        actions = replay.actions()

        new_observations = [list(obs) for obs in zip(*observations)]
        new_actions = [list(act) for act in zip(*actions)]

        for i, (cur_obs, prev_act) in enumerate(zip(new_observations, new_actions)):
            [state_h , state_c] = cur_state

            next_internal_state, sampled_actions, log_probs = \
                self.policy.sample_step(cur_obs, prev_act, (state_h, state_c))
            
            cur_state = next_internal_state
                                        
            all_log_probs.append(log_probs)
            all_states.append(cur_state)

        all_states = all_states[:-1]
        all_log_probs = all_log_probs[:-1]

        return all_log_probs, all_states

    #?
    def get_episode_reward_lengths(self, rewards, pads, terminated):
        self.total_rewards = np.sum(np.array(rewards[self.start_id:]) *
                                    (1 - np.array(pads[self.start_id:])), axis=0)
        self.episode_running_rewards *= 1 - self.start_episode
        self.episode_running_lengths *= 1 - self.start_episode

        self.episode_running_rewards += np.sum(np.array(rewards[self.start_id:]) *
                                               (1 - np.array(pads[self.start_id:])), axis=0)
        self.episode_running_lengths += np.sum(1 -
                                               np.array(pads[self.start_id:]), axis=0)

        # set next starting episodes
        self.start_episode = np.logical_or(terminated,
                                           self.step_count >= self.cutoff_agent)

        self.episode_rewards.extend(
            self.episode_running_rewards[self.start_episode].tolist())
        self.episode_lengths.extend(
            self.episode_running_lengths[self.start_episode].tolist())
        # *这里只取最后面的100个episode->计算lambda

        self.episode_rewards = self.episode_rewards[-100:]
        self.episode_lengths = self.episode_lengths[-100:]

    def train(self, cur_step):

        def to_full_tensor(value):
            value = [torch.stack(v) for v in value]
            return value
        
        def states_to_full_tensor(states):
            states = [torch.squeeze(torch.stack(state, axis=0)) for state in states]
            states = torch.stack(states, axis=0)
            states = torch.transpose(states, 1, 2)
            return states

        # TODO 1 move all data in tensor!
        # TODO 1.1 Tensor in cpu
        # TODO 1.2 remove all append
        self.cur_step = cur_step
        
        # ************** sampling *************
        # sample from policy
        (initial_state, observations, actions, rewards,
            terminated, pads) = self.sampling_episode()
        
        # ************** sampling *************
        sampled_episodes = Replay((initial_state, observations, actions, rewards,
            terminated, pads))        

        # ************** calc statistics *************
        log_probs, all_states = self.get_log_probs_and_states_from_policy(sampled_episodes)

        values = self.critic.get_values(to_full_tensor(sampled_episodes.observations()), 
                                        to_full_tensor(sampled_episodes.actions()), 
                                        states_to_full_tensor(all_states))
        
        loss = self.loss_func.get_loss(values, sampled_episodes, log_probs)
        # ************** calc statistics *************
        

        # ************** back forwarding *************
        loss.backward()
        self.optimizer.step()
        # self.soft_update_of_target_network()
        self.optimizer.zero_grad()
        # ************** back forwarding *************

        self.episode_to_buffer(sampled_episodes)
        replay_batch, _ = self.select_from_buffer(self.replay_batch_size)
        buffered_episodes = Replay(replay_batch)

        if not buffered_episodes.empty:

            # ************** not checked *************
            # episode_rewards = np.array(self.episode_rewards)
            # episode_lengths = np.array(self.episode_lengths)

            # direction func to update lambda
            # not necessary to have this in PCL
            # self.objective.update_lambda(self.find_best_eps_lambda(
            #     np.array(self.episode_rewards)[-20:], np.array(self.episode_lengths)[-20:]))
            # ************** not checked *************
            

            # ************** sampling *************
            # sample from buffer
            (initial_state, observations, actions, rewards,
                terminated, pads) = replay_batch
            # ************** sampling *************
            

            # ************** calc statistics *************
            log_probs, all_states = self.get_log_probs_and_states_from_policy(buffered_episodes)

            values = self.critic.get_values(to_full_tensor(buffered_episodes.observations()), 
                                            to_full_tensor(buffered_episodes.actions()), 
                                            states_to_full_tensor(all_states))

            loss = self.loss_func.get_loss(values, sampled_episodes, log_probs)
            # ************** calc statistics *************
            

            # ************** back forwarding *************
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # self.soft_update_of_target_network()
            # ************** back forwarding *************

            return loss, self.total_rewards, self.episode_rewards

    def find_best_eps_lambda(self, rewards, lengths):
        """Find the best lambda given a desired epsilon = FLAGS.max_divergence."""
        # perhaps not the best way to do this
        # can use numpy to cal lambda, because lambda require no gradient
        desired_div = self.objective.max_divergence * np.mean(lengths)

        def calc_divergence(eps_lambda):
            max_reward = np.max(rewards)
            logz = (max_reward / eps_lambda +
                    np.log(np.mean(np.exp((rewards - max_reward) / eps_lambda))))
            exprr = np.mean(np.exp(rewards / eps_lambda - logz) *
                            rewards / eps_lambda)
            return exprr - logz

        left = 0.0
        right = 1000.0

        if len(rewards) <= 8:
            return (left + right) / 2

        num_iter = max(4, 1 + int(np.log((right - left) / 0.1) / np.log(2.0)))
        for _ in xrange(num_iter):
            mid = (left + right) / 2
            cur_div = calc_divergence(mid)
            if cur_div > desired_div:
                left = mid
            else:
                right = mid

        return (left + right) / 2

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def initial_internal_state(self):
        return np.zeros(128), np.zeros(128)
    
    def select_from_buffer(self, batch_size):
        """Sample a batch of episodes from the replay buffer."""
        if self.replay_buffer is None or len(self.replay_buffer) < 1 * batch_size:
            return None, None

        desired_count = batch_size * self.max_step
        # in the case of batch_by_steps, we sample larger and larger
        # amounts from the replay buffer until we have enough steps.
        while True:
            batch_size = min(batch_size, len(self.replay_buffer))
            episodes, probs = self.replay_buffer.get_batch(batch_size)
            count = sum(len(ep[-2]) for ep in episodes)
            if count >= desired_count or not self.FLAG_batch_by_steps:
                break
            if batch_size == len(self.replay_buffer):
                return None, None
            batch_size *= 1.2

        return (Replay.convert_to_batched_episodes(episodes), probs)
    
    def episode_to_buffer(self, replay):
        """Add batch of episodes to replay buffer."""
        if self.replay_buffer is None:
            return

        rewards = np.array(replay.rewards())
        pads = np.array(replay.pads())
        total_rewards = np.sum(rewards * (1 - pads), axis=0)

        episodes = replay.to_episodes()


        priorities = (total_rewards if self.prioritize_by == 'reward'
                      else self.cur_step)

        if not self.FLAG_unify_episodes or self.all_new_ep:
            self.last_idxs = self.replay_buffer.add(
                episodes, priorities)
        else:
            # If we are unifying episodes, we attempt to
            # keep them unified in the replay buffer.
            # The first episode sampled in the current batch is a
            # continuation of the last episode from the previous batch
            self.replay_buffer.add(
                episodes[:1], priorities, self.last_idxs[-1:])
            if len(episodes) > 1:
                self.replay_buffer.add(episodes[1:], priorities)