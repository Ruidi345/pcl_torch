import math

import numpy as np
import tensorflow as tf
from six.moves import xrange

import carla_env
import env_spec
import gym_wrapper
import model
import objective
import replay_buffer



class Controller():
    def __init__(self,batch_size=400,
                env_name = 'Copy-v0',
                env = None,
                train_objective = 'trust_pcl',
                unify_episodes = False,
                use_online_batch = True,
                max_step = 200,
                cutoff_agent = 0,
                batch_by_steps = False,
                input_time_step = True,
                replay_batch_size = -1,
                target_network_lag = 0.95,
                clip_norm = 20,
                use_target_values=False,
                # max_divergence = 0.5,
                prioritize_by = 'rewards',
                online_model = None,
                target_model = None,
                objective = None,
                the_replay_buffer = None,
                tau = 0.1,
                total_steps = 10000,
                start_id = 0,                
                cur_step = 0,
                validation_frequency = 250,
                decay_tau=False,
                get_buffer_seeds=None,
                value_opt=False,
                train = True):
        
        self.batch_size = batch_size

        self.env_name = env_name
        self.env = env
        self.env_spec = env_spec.EnvSpec(self.env.get_one())


        self.train_objective = train_objective
        # self.train_objective = 'pcl'
        self.FLAG_use_trust_pcl = self.train_objective == 'trust_pcl' or self.train_objective == 'trust_spcl'
        self.replay_buffer = the_replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        self.objective = objective
        self.prioritize_by = prioritize_by
        self.tau_decay = decay_tau
        self.total_steps = total_steps
        self.next_tau_update = int(self.total_steps/10)
        self.tau = tau

        self.value_opt = value_opt

        self.cur_step = cur_step
        self.FLAG_unify_episodes = unify_episodes
        self.FLAG_use_online_batch = use_online_batch
        self.max_step = max_step
        self.start_id = start_id
        self.cutoff_agent = cutoff_agent
        self.FLAG_batch_by_steps = batch_by_steps
        self.FLAG_input_time_step = input_time_step
        self.target_network_lag = target_network_lag
        self.clip_norm = clip_norm        

        self.episode_running_rewards = np.zeros(len(self.env))
        self.episode_running_lengths = np.zeros(len(self.env))
        self.step_count = np.array([0] * len(self.env))
        self.start_episode = np.array([True] * len(self.env))
        self.episode_rewards = []
        self.episode_lengths = []
        self.replay_batch_size = self.batch_size if replay_batch_size<0 else replay_batch_size

        self.internal_state = np.array([self.initial_internal_state()] *
                                       len(self.env))
        self.internal_state_t = np.array([self.initial_internal_state()] *
                                       len(self.env))
        self.last_obs = self.env_spec.initial_obs(len(self.env))
        self.last_act = self.env_spec.initial_act(len(self.env))
        self.last_pad = np.zeros(len(self.env))

        self.start_loss = True
        self.show_step = False
        self.validation_frequency = validation_frequency
        if get_buffer_seeds is not None:
            self.seed_replay_buffer(get_buffer_seeds())

        self.if_train = train
        

    def set_writer(self, writer):
        self.writer = writer
        self.objective.set_writer(self.writer, self.validation_frequency)

    def initial_internal_state(self):
        return tf.zeros(128), tf.zeros(128)

    def cut_final_values(self, values, terminated):
        return values[:-1, :], values[-1, :] * (1 - terminated)

    def train(self, cur_step):
        self.cur_step = cur_step
        self.objective.set_step(self.cur_step)

        if self.cur_step == 2:
            for _ in range(100):
                self.update_model_weights()
        self.update_model_weights()

        loss = tf.zeros([1], tf.float32)
        with tf.GradientTape(persistent=False) as tape:

            (initial_state, observations, actions, rewards,
             terminated, pads) = self.sampling_episode()

        
            if not self.if_train:
                return None, None, None

            if self.FLAG_use_online_batch:
                values_, final_values, all_log_probs, logits, reg_inputs, entropies = self.episode_processing(initial_state, \
                    observations, actions, terminated)
                if self.FLAG_use_trust_pcl:
                    target_values_, target_final_values, target_all_log_probs, target_logits, _ , _ = \
                        self.episode_processing(initial_state, \
                        observations, actions, terminated, use_target=True)
                else:
                    target_values_, target_final_values, target_all_log_probs, target_logits=\
                        [], [], [], []

                loss, reg_target = self.objective.grad(rewards, pads, values_, final_values, all_log_probs, logits,\
                                        target_values_, target_final_values, target_all_log_probs, target_logits, entropies, actions)

                            

        if self.FLAG_use_online_batch:
            grads = tape.gradient(loss, self.online_model.trainable_variables)
            # grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
            self.objective.optimizer.apply_gradients(
                zip(grads, self.online_model.trainable_variables))

            # if self.value_opt:
            #     values_, final_values, all_log_probs, logits = self.episode_processing(initial_state, \
            #         observations, actions, terminated)
            #     self.value_optimize()

        self.episode_to_buffer(initial_state, observations, actions,
                               rewards, terminated, pads)


        replay_batch, _ = self.select_from_buffer(self.replay_batch_size)

        if replay_batch is not None:
            with tf.GradientTape(persistent=False) as tape:

                if self.FLAG_use_trust_pcl:
                    episode_rewards = np.array(self.episode_rewards)
                    episode_lengths = np.array(self.episode_lengths)

                    self.objective.update_lambda(self.find_best_eps_lambda(
                        episode_rewards[-20:], episode_lengths[-20:]))

                (initial_state, observations, actions, rewards,
                 terminated, pads) = replay_batch
                # initial_state   [batch_size, internal_dim]
                # observations    [time_length, batch_size]
                # actions         [time_length, batch_size]
                # rewards         [time_length-1, batch_size]
                # terminated      [batch_size]
                # pads            [time_length, batch_size]
                values_, final_values, all_log_probs, logits, reg_inputs, entropies = self.episode_processing(initial_state, \
                    observations, actions, terminated)
                if self.FLAG_use_trust_pcl:
                    target_values_, target_final_values, target_all_log_probs, target_logits, _, _ = \
                        self.episode_processing(initial_state, \
                        observations, actions, terminated, use_target=True)
                else:
                    target_values_, target_final_values, target_all_log_probs, target_logits, reg_inputs=\
                        [], [], [], [], []

                loss, reg_target = self.objective.grad(rewards, pads, values_, final_values, all_log_probs, logits,\
                                        target_values_, target_final_values, target_all_log_probs, target_logits, entropies, actions)
            grads = tape.gradient(loss, self.online_model.trainable_variables)
            var = self.online_model.trainable_variables
            # grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
            self.objective.optimizer.apply_gradients(
                zip(grads, self.online_model.trainable_variables))


        self.update_tau()

        return loss.numpy(), self.total_rewards, self.episode_rewards

    def evaluate(self, cur_step):
        with tf.GradientTape(persistent=False) as tape:
            (initial_state, observations, actions, rewards,
                terminated, pads) = self.sampling_episode()
        return self.total_rewards, self.episode_rewards

    def episode_processing(self, initial_state, observations, actions, terminated, use_target=False):
        # 在源码的policy.multi_step()中可以看出函数参数internal_states为initial_state

        model = self.target_model if use_target else self.online_model

        all_states, all_log_probs, logits, entropies = model.multi_step(observations, initial_state, actions)
        values, reg_inputs = model.get_values(observations, actions, all_states)

        if len(values.shape) == 3:
            values_cutted = values[:-1, :, :]
            final_value = values[-1, :, 0] * (1 - terminated)
        else:
            values_cutted = values[:-1, :]
            final_value = values[-1, :] * (1 - terminated)

        return values_cutted, final_value, all_log_probs, logits, reg_inputs, entropies

    def sampling_episode(self, greedy=False):
        total_steps = 0
        episodes = []

        # 此while loop获得batch size数量的episode
        while total_steps < self.max_step * len(self.env):
            # episodes形式的step length的数据
            initial_state, observations, actions, rewards, pads = self._sampling_episode()

            # *************** data processing ***************
            self.online_model.reset()
            # if self.use_trust_pcl:
            #     self.target_model.reset()

            observations = list(zip(*observations))
            actions = list(zip(*actions))

            terminated = np.array(self.env.dones)
            episodes.extend(self.convert_from_batched_episodes(
                initial_state, observations, actions, rewards,
                terminated, pads))

            total_steps += np.sum(1 - np.array(pads))

            self.get_episode_reward_lengths(rewards, pads, terminated)

            if not self.FLAG_batch_by_steps:
                return (initial_state,
                        observations, actions, rewards, terminated, pads)
                    
        # *************** data processing ***************

        return self.convert_to_batched_episodes(episodes)

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



    def _sampling_episode(self):
        all_obs, all_act, all_pad, rewards, all_states = [], [], [], [], []
        step = 0

        #! self.start_episode
        obs_after_reset = self.env.reset_if(self.start_episode)
        # obs_after_reset: conti:[[np.array()]                   ]
        # obs_after_reset: discr:[[int],       [int], ..., [int] ]
        # print("obs: ", self.start_episode, obs_after_reset)

        for i, obs in enumerate(obs_after_reset):
            # obs: [np.array()] 
            # i: batch size
            if obs is not None:
                self.step_count[i] = 0
                self.internal_state[i] = self.initial_internal_state()
                self.internal_state_t[i] = self.initial_internal_state()
                for j in xrange(len(self.env_spec.obs_dims)):
                    # self.last_obs.shape=(obs_dim, batch_size)

                    self.last_obs[j][i] = obs[j]
                for _ in xrange(len(self.env_spec.act_dims)):
                    # self.last_act[j][i] = -1
                    self.last_pad[i] = 0
                # self.last_obs = [self.last_obs]


        # 记录net states的初始化
        initial_state_h = tf.convert_to_tensor(
            value=np.squeeze(self.internal_state[:, 0:1, :], axis=1))
        initial_state_c = tf.convert_to_tensor(
            value=np.squeeze(self.internal_state[:, 1:, :], axis=1))

        self.internal_state = [initial_state_h, initial_state_c]

        initial_state = self.internal_state

        # all_act添加上最初的dummy action(-1),使得最后输出act的长度多一个
        all_act.append(self.last_act)

        # 每循环一次，推进的是episode中的一个时间步
        while not self.env.all_done():
            self.step_count += 1 - np.array(self.env.dones)
            next_internal_state, sampled_actions, logits, log_probs, entropy, self_kl = \
                self.online_model.single_step(
                    self.last_obs, self.last_act, self.internal_state)
            if self.cur_step == 1 and self.FLAG_use_trust_pcl:
                _, _, _, _, _, _ = self.target_model.single_step(\
                                        self.last_obs, self.last_act, self.internal_state)     
               
            env_actions = self.env_spec.convert_actions_to_env(sampled_actions)

            all_act.append(sampled_actions)  

            next_obs, reward, next_dones, _ = self.env.step(env_actions)

            all_obs.append(self.last_obs)  
            all_pad.append(self.last_pad)  
            rewards.append(reward)

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

    def episode_to_buffer(self, initial_state,
                          observations, actions, rewards,
                          terminated, pads):
        """Add batch of episodes to replay buffer."""
        if self.replay_buffer is None:
            return

        rewards = np.array(rewards)
        pads = np.array(pads)
        total_rewards = np.sum(rewards * (1 - pads), axis=0)

        episodes = self.convert_from_batched_episodes(
            initial_state, observations, actions, rewards,
            terminated, pads)

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

        return (self.convert_to_batched_episodes(episodes), probs)

    def convert_from_batched_episodes(
            self, initial_state, observations, actions, rewards,
            terminated, pads):
        """Convert time-major batch of episodes to batch-major list of episodes."""
        rewards = np.array(rewards)
        pads = np.array(pads)
        observations = [np.array(obs) for obs in observations]
        actions = tf.stack(actions)
        actions = tf.squeeze(tf.stack(actions), axis=0)

        total_rewards = np.sum(rewards * (1 - pads), axis=0)
        total_length = np.sum(1 - pads, axis=0).astype('int32')

        episodes = []
        num_episodes = rewards.shape[1]

        for i in xrange(num_episodes):

            length = total_length[i]
            ep_initial = [tf.expand_dims(state[i], 0) for state in initial_state]
            ep_obs = [obs[:length + 1, i, ...] for obs in observations]
            ep_act = [actions[:length + 1, i, ...]]
            ep_rewards = rewards[:length, i]

            episodes.append(
                [ep_initial, ep_obs, ep_act, ep_rewards, terminated[i]])

        return episodes

    def convert_to_batched_episodes(self, episodes, max_length=None):
        """Convert batch-major list of episodes to time-major batch of episodes."""
        #* 解压函数
        #* sampling episode返回时会调用，要求states的shape包含batch_size
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

        (repaly_initial, observations, actions, rewards,
         terminated, pads) = zip(*new_episodes)
        observations = [np.swapaxes(obs, 0, 1)
                        for obs in zip(*observations)]
        actions = [np.swapaxes(act, 0, 1)
                   for act in zip(*actions)]
        rewards = np.transpose(rewards)
        pads = np.transpose(pads)

        repaly_initial = np.array(repaly_initial)
        if len(repaly_initial.shape) != 3:
            repaly_initial = tf.squeeze(repaly_initial, 2)
        initial_state_h = tf.convert_to_tensor(
            value=tf.squeeze(repaly_initial[:, 0:1, :], 1))

        initial_state_c = tf.convert_to_tensor(
            value=tf.squeeze(repaly_initial[:, 1:, :], 1))
        initial = [initial_state_h, initial_state_c]
        terminated = np.array(terminated)

        return (initial, observations, actions, rewards, terminated, pads)

    def update_model_weights(self):
        ratio = self.target_network_lag  # 0.95

        if self.target_model is None or not self.if_train:
            return

        for online_var, target_var in zip(self.online_model.trainable_variables, self.target_model.trainable_variables):
            target_var.assign(ratio * target_var + (1-ratio) * online_var)

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

    def update_tau(self):
        # initial_tau = 1
        decay_rate = 0.8
        decay_steps = self.total_steps / 10

        def decayed_learning_rate(step):
            return decay_rate ** (step / decay_steps)

        def decayed_learning_rate_cos(step, end_precentage):
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
            decayed = (1 - end_precentage) * cosine_decay + end_precentage
            return decayed

        if self.tau_decay:
            if self.cur_step > self.next_tau_update:
                tau_coeff = decayed_learning_rate(self.cur_step)
                self.objective.tau = self.tau *tau_coeff
                if hasattr(self.online_model, "tau"):
                    self.online_model.tau = self.tau * tau_coeff
                if self.target_model is not None and hasattr(self.target_model, "tau"):
                    self.target_model.tau = self.tau * tau_coeff
                self.next_tau_update = self.cur_step + decay_steps
        else:
            return

    def value_optimize(self, reg_input, reg_weight, old_values, reg_target):
        self.mix_frac = 0
        old_values = tf.reshape(old_values[:-1, :], [-1])
        intended_values = reg_target * self.mix_frac + old_values * (1 - self.mix_frac)

        # taken from rllab
        reg_coeff = 1e-5
        for _ in range(5):
            best_fit_weight = np.linalg.lstsq(
                reg_input.T.dot(reg_input) +
                reg_coeff * np.identity(reg_input.shape[1]),
                reg_input.T.dot(intended_values))[0]
            if not np.any(np.isnan(best_fit_weight)):
                break
            reg_coeff *= 10

        if len(best_fit_weight.shape) == 1:
            best_fit_weight = np.expand_dims(best_fit_weight, -1)

        # sess.run(self.update_regression_weight,
        #         feed_dict={self.new_regression_weight: best_fit_weight})
        for var in self.online_model.trainable_variables:
            if var.name == 'online_value_wrapper/kernel:0':
                var.assign(best_fit_weight)

    def seed_replay_buffer(self, episodes):
        """Seed the replay buffer with some episodes."""
        if self.replay_buffer is None:
            return

        # just need to add initial state
        for i in xrange(len(episodes)):
            init_state = self.initial_internal_state()
            init_states = tf.expand_dims(init_state[0], 0),tf.expand_dims(init_state[1], 0)
            episodes[i] = [init_states] + episodes[i]

        self.replay_buffer.seed_buffer(episodes)

        

