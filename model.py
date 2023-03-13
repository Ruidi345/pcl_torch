import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

import gym_wrapper
import env_spec

dev_initializers = tf.initializers.truncated_normal(stddev=0.01, seed=None)
vec_initializers = tf.initializers.constant(value=0)
# dev_initializers = tf.initializers.truncated_normal(mean=0.0, stddev=0.01, seed=None, dtype=tf.dtypes.float32)

def spmax_tau(logits):
  batch_size = tf.shape(input=logits)[0]
  num_actions = tf.shape(input=logits)[1]

  z = logits

  z_sorted, _ = tf.nn.top_k(z, k=num_actions)

  z_cumsum = tf.cumsum(z_sorted, axis=1)
  k = tf.range(1, tf.cast(num_actions, logits.dtype) + 1, dtype=logits.dtype)
  z_check = 1 + k * z_sorted > z_cumsum

  k_z = tf.reduce_sum(input_tensor=tf.cast(z_check, tf.int32), axis=1)

  indices = tf.stack([tf.range(0, batch_size), k_z - 1], axis=1)
  tau_sum = tf.gather_nd(z_cumsum, indices)
  tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

  return tau_z

class PolicyEncoder(Layer):

    def __init__(self, env_spec, internal_dim, input_prev_actions, name='online'):
        super(PolicyEncoder, self).__init__()
        # TODO may change this later
        self.env_spec = env_spec
        self.input_dim = internal_dim
        self.model_name = name
        self.input_prev_actions = input_prev_actions

        self.layer_vars = {}
        self.bias = self.add_weight(
            shape=(self.input_dim,), name='input_bias', initializer=vec_initializers)

        for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
            self.layer_vars[name+'w_state%d' % i] = self.add_weight(shape=(
                obs_dim, self.input_dim,), initializer=dev_initializers, name=name+'w_state%d' % i)

        if self.input_prev_actions:
            if self.env_spec.combine_actions:
                for i, action_dim in enumerate(self.env_spec.orig_act_dims):
                    self.layer_vars[name+'w_prev_action%d' % i] = self.add_weight(shape=(
                        action_dim, self.input_dim,), initializer=dev_initializers, name=name+'w_prev_action%d' % i)
            else:
                for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
                    self.layer_vars[name+'w_prev_action%d' % i] = self.add_weight(shape=(
                        act_dim, self.input_dim,), 
                        initializer=dev_initializers, 
                        name=name+'w_prev_action%d' % i)

    
    def call(self, observations, prev_actions):

        batch_size = tf.shape(prev_actions[0])[0]
        cell_input = tf.nn.bias_add(
            tf.zeros([batch_size, self.input_dim]), self.bias)

        # *****处理放入LTSM cell的数据*****
        # self.env_spec.obs_dims_and_types=((6,0),) len=100
        # obs_dim=6, obs_type=0

        # observation[0].shape = [batchsize, obs_dim]
        # observation[0].shape = [time_length*batchsize, obs_dim]
        for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
            if self.env_spec.is_discrete(obs_type):
                # tf.one_hot(obs[i], obs_dim)将obs拓展成[100, obs_dim]的矩阵
                # obs_i = tf.cast(obs[i], tf.int32)
                one_hot_obs = tf.one_hot(
                    tf.cast(observations[i], tf.int32), obs_dim)
                one_hot_obs = tf.cast(one_hot_obs, tf.float32)
                cell_input += tf.linalg.matmul(one_hot_obs,
                                               self.layer_vars[self.model_name+'w_state%d' % i])
            elif self.env_spec.is_box(obs_type):
                observations = tf.cast(observations, tf.float32)
                cell_input += tf.linalg.matmul(
                    observations[i], self.layer_vars[self.model_name+'w_state%d' % i])
            else:
                assert False

        if self.input_prev_actions:
            # if True:
            if self.env_spec.combine_actions:  # TODO(ofir): clean this up
                # prev_action:[(100)]
                # prev_action[0].shape= [time_length * batch_size]
                prev_action = prev_actions[0]
                # for里面走三次循环，action_dim
                # 包含对action的解码操作，将一维数据升为三维(取余)
                for i, action_dim in enumerate(self.env_spec.orig_act_dims):
                    act = tf.math.floormod(prev_action, action_dim)
                    act = tf.cast(act, tf.int32)
                    cell_input += tf.linalg.matmul(tf.one_hot(
                        act, action_dim), self.layer_vars[self.model_name+'w_prev_action%d' % i])
                    prev_action = tf.cast(prev_action / action_dim, tf.int32)
            else:
                for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
                    if self.env_spec.is_discrete(act_type):
                        cell_input += tf.linalg.matmul(tf.one_hot(
                            prev_actions[i], act_dim), self.layer_vars[self.model_name+'w_prev_action%d' % i])
                    elif self.env_spec.is_box(act_type):
                        cell_input += tf.linalg.matmul(
                            prev_actions[i], self.layer_vars[self.model_name+'w_prev_action%d' % i])
                    else:
                        assert False

        return cell_input


class PCLModel(Model):
    def __init__(self, env_spec, name='online_'):
        super(PCLModel, self).__init__()
        self.env_spec = env_spec



    @property
    def output_dim(self):
        return self.env_spec.total_sampling_act_dim

    # sample_step()
    def call(self, obs, act, state_h, state_c,):
        raise NotImplementedError()

    def forward(self, states):
        raise NotImplementedError()

    # input: self.last_obs, self.internal_state, self.last_act
    # output: next_internal_state, sampled_actions, logits, log_probs, entropy, self_kl
    def single_step(self, last_obs, last_act, internal_state, cur_act=None):
        raise NotImplementedError()

    def multi_step(self, all_obs, initial_state, all_actions):
        raise NotImplementedError()

    def process_net_outputs(self, output, cur_act):
        """Sample all actions in a batch given output of core network."""
        # actions len=1，其元素为len=100的array
        sampled_actions = []
        logits = []
        log_probs = []
        entropy = []
        self_kl = []

        start_idx = 0
        # self.env_spec.act_dims_and_types=((20,0),)
        # act_dim=20, act_type=0
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
            # sampling_dim =20,这里是离散sampling空间
            sampling_dim = self.env_spec.sampling_dim(act_dim, act_type)

            # TODO continue space, may need trainable variable for sampling from a distribution
            if self.fixed_std and self.env_spec.is_box(act_type):
                act_logits = output[:, start_idx:start_idx + act_dim]

                # log_std = tf.compat.v1.get_variable(
                #     'std%d' % i, [1, sampling_dim // 2])
                # test_var = self.log_std[self.model_name+'std%d' % i]
                # fix standard deviations to variable
                # 0 * act_logits:let the log_std(shape:[1, sampling_dim // 2])
                # to be expanded to the same shape as act_logits
                act_logits = tf.concat(
                    [act_logits,
                     1e-6 + tf.exp(self.log_std[self.model_name+'std%d' % i]) + 0 * act_logits], 1)
            else:
                act_logits = output[:, start_idx:start_idx + sampling_dim]

            # output for observations at time step t
            # act_logits = output[:, start_idx:start_idx + sampling_dim]
            #
            if cur_act is None:
                act = self.sample_action(act_logits, sampling_dim,
                                         act_dim, act_type)
                sampled_actions.append(act)

            else:
                act = cur_act

            ent = self.entropy(act_logits, sampling_dim, act_dim, act_type)
            kl = self.self_kl(act_logits, sampling_dim, act_dim, act_type)
            # shape (100,)
            act_log_prob = self.log_prob_action(
                act, act_logits,
                sampling_dim, act_dim, act_type)

            if self.tsaills:
                act_logits = act_logits / (self.k * self.q * self.tau)

            logits.append(act_logits)
            self_kl.append(kl)
            entropy.append(ent)
            log_probs.append(act_log_prob)

            start_idx += sampling_dim

        assert start_idx == self.env_spec.total_sampling_act_dim

        return sampled_actions, logits, log_probs, entropy, self_kl

    def sample_action(self, logits, sampling_dim,
                      act_dim, act_type, greedy=False):
        # sourcery skip: merge-duplicate-blocks
        """Sample an action from a distribution."""
        if self.env_spec.is_discrete(act_type):
            if self.sample_policy == 'greedy':
                act = tf.argmax(input=logits, axis=1)
            elif self.sample_policy == 'spmax':
                probs = tf.nn.relu(logits - tf.expand_dims(spmax_tau(logits), 1))
                dist = tf.compat.v1.distributions.Categorical(probs=probs)
                act = dist.sample([])
            elif self.sample_policy == 'softmax':
                act = tf.reshape(tf.random.categorical(
                    logits=logits, num_samples=1), [-1])
            else:
                assert False
        elif self.env_spec.is_box(act_type):
            # TODO continue space
            # assert False
            div = int(sampling_dim/2)
            means = logits[:, :div]
            std = logits[:, div:]
            if greedy:
                act = means
            else:
                batch_size = tf.shape(logits)[0]
                act = means + std * tf.random.normal([batch_size, act_dim])
        else:
            assert False

        return act

    def self_kl(self, logits,
                sampling_dim, act_dim, act_type):
        # sourcery skip: merge-duplicate-blocks
        """Calculate KL of distribution with itself.

        Used layer only for the gradients.
        """

        if self.env_spec.is_discrete(act_type):
            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)
            self_kl = tf.reduce_sum(
                input_tensor=tf.stop_gradient(probs) *
                (tf.stop_gradient(log_probs) - log_probs), axis=-1)
        elif self.env_spec.is_box(act_type):
            # assert False
            div = int(sampling_dim/2)
            means = logits[:, :div]
            std = logits[:, div:]
            my_means = tf.stop_gradient(means)
            my_std = tf.stop_gradient(std)
            self_kl = tf.reduce_sum(
                tf.math.log(std / my_std) +
                (tf.square(my_std) + tf.square(my_means - means)) /
                (2.0 * tf.square(std)) - 0.5,
                -1)
        else:
            assert False

        return self_kl

    def entropy(self, logits,
                sampling_dim, act_dim, act_type):
        # sourcery skip: merge-duplicate-blocks
        """Calculate entropy of distribution."""
        # sampling_dim=20, act_dim=20, act_type=0
        if self.env_spec.is_discrete(act_type):
            entropy = tf.reduce_sum(
                input_tensor=-tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1)
        elif self.env_spec.is_box(act_type):
            # assert False
            div = int(sampling_dim/2)
            means = logits[:, :div]
            std = logits[:, div:]
            entropy = tf.reduce_sum(
                0.5 * (1 + tf.math.log(2 * np.pi * tf.square(std))), -1)
        else:
            assert False

        return entropy

    def log_prob_action(self, action, logits,
                        sampling_dim, act_dim, act_type):
        # sourcery skip: merge-duplicate-blocks
        """Calculate log-prob of action sampled from distribution."""
        # action, len=100
        if self.env_spec.is_discrete(act_type):
            if self.tsaills:
            # tf.one_hot(action, act_dim).shape=(100,20)
            # tf.nn.log_softmax(logits).shape=(100,20)
                probs = tf.nn.relu(logits - tf.expand_dims(spmax_tau(logits), 1))
                act_log_prob = tf.reduce_sum(
                input_tensor=tf.one_hot(action, act_dim) * tf.math.log(1e-6 + probs), axis=-1)
            else:
                action = tf.cast(action, tf.int32)
                act_log_prob = tf.reduce_sum(
                    input_tensor=tf.one_hot(action, act_dim) * tf.nn.log_softmax(logits), axis=-1)
        elif self.env_spec.is_box(act_type):
            # assert False
            div = int(sampling_dim/2)
            means = logits[:, :div]
            std = logits[:, div:]
            act_log_prob = (- 0.5 * tf.math.log(2 * np.pi * tf.square(std))
                            - 0.5 * tf.square(action - means) / tf.square(std))
            act_log_prob = tf.reduce_sum(act_log_prob, -1)
        else:
            assert False

        return act_log_prob

    def get_values(self, all_obs, all_actions, all_states):
        # all_states.shape=[time_length, batch_size, ...]
        all_states = tf.convert_to_tensor(value=all_states)
        # batch_size = tf.shape(all_states)[2]
        batch_size = len(all_obs[0][1])
        # time_length = tf.shape(all_states)[0]
        time_length = len(all_obs[0])
        internal_policy_dim = tf.shape(all_states)[3]

        (time_step, reshaped_obs, reshaped_prev_act,
         reshaped_internal_policy_states) = self.reshape_batched_inputs(
            all_obs, all_actions, all_states)

        input_dim, inputs = self.get_inputs(
            time_step, reshaped_obs, reshaped_prev_act,
            reshaped_internal_policy_states)

        # for normal pcl:
        # value.shape = [time_length* batch_size, 1]
        values, reg_inputs = self.forward(inputs)

        # for sparse pcl
        if len(values.shape) == 3:
            values = tf.reshape(values, [-1, time_length, batch_size])
            values = tf.transpose(values, perm=[1,2,0])
        else:
            values = tf.reshape(values, [time_length, batch_size])

        reg_inputs = reg_inputs[:-batch_size]
        
        return values, reg_inputs

    def reshape_batched_inputs(self, all_obs, all_actions,
                               internal_policy_states):
        """Reshape inputs from [time_length, batch_size, ...] to
        [time_length * batch_size, ...].

        This allows for computing the value estimate in one go.
        """
        batch_size = tf.shape(internal_policy_states)[2]
        time_length = tf.shape(internal_policy_states)[0]
        internal_policy_dim = tf.shape(internal_policy_states)[3]
        reshaped_internal_policy_dim = internal_policy_dim * 2

        reshaped_obs = []
        # # self.env_spec.obs_dims_and_types=((6,0),) len=100
        # # obs_dim=6, obs_type=0
        for obs, (obs_dim, obs_type) in zip(all_obs, self.env_spec.obs_dims_and_types):
            if self.env_spec.is_discrete(obs_type):
                reshaped_obs.append(tf.reshape(
                    obs, [time_length * batch_size]))
            elif self.env_spec.is_box(obs_type):
                reshaped_obs.append(tf.reshape(
                    obs, [time_length * batch_size, obs_dim]))

        reshaped_prev_act = []
        # # self.env_spec.act_dims_and_types=((20,0),)
        # # act_dim=20, act_type=0
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
            prev_act = all_actions[i]
            if self.env_spec.is_discrete(act_type):
                reshaped_prev_act.append(
                    tf.reshape(prev_act, [time_length * batch_size]))
            elif self.env_spec.is_box(act_type):
                reshaped_prev_act.append(
                    tf.reshape(prev_act, [time_length * batch_size, act_dim]))
        reshaped_internal_policy_states = tf.reshape(
            internal_policy_states,
            [time_length * batch_size, reshaped_internal_policy_dim])


        time_step = (float(self.input_time_step) * tf.expand_dims(
            tf.cast(tf.range(time_length * batch_size) /
                    batch_size, dtype=tf.float32), -1))

        return (time_step, reshaped_obs, reshaped_prev_act,
                reshaped_internal_policy_states)

    def get_inputs(self, time_step, obs, prev_actions,
                   internal_policy_states):
        """Get inputs to network as single tensor."""
        inputs = [tf.ones_like(time_step)]
        input_dim = 1

        if not self.input_policy_state:
            for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
                if self.env_spec.is_discrete(obs_type):
                    inputs.append(
                        tf.one_hot(obs[i], obs_dim))
                    input_dim += obs_dim
                elif self.env_spec.is_box(obs_type):
                    cur_obs = tf.cast(obs[i], tf.float32)
                    inputs.append(cur_obs)
                    inputs.append(cur_obs ** 2)
                    input_dim += obs_dim * 2
                else:
                    assert False

            # LSTM:self.input_prev_actions = False
            if self.input_prev_actions:
                for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
                    if self.env_spec.is_discrete(act_type):
                        inputs.append(
                            tf.one_hot(prev_actions[i], act_dim))
                        input_dim += act_dim
                    elif self.env_spec.is_box(act_type):
                        inputs.append(prev_actions[i])
                        input_dim += act_dim
                    else:
                        assert False

        if self.input_policy_state:
            inputs.append(internal_policy_states)
            input_dim += internal_policy_states.shape[1]

        # LSTM:self.input_time_step = False
        if self.input_time_step:
            scaled_time = 0.01 * time_step
            inputs.extend([scaled_time, scaled_time ** 2, scaled_time ** 3])
            input_dim += 3

        # input.shape=[time_length * batch_size, self.internal_policy_dim+1]
        return input_dim, tf.concat(inputs, 1)


class RNNPCLModel(PCLModel):
    def __init__(self, env_spec,  input_time_step=False,
                 input_policy_state=True,
                 input_prev_actions=False,
                 fixed_std=False,
                 value_hidden_layers=2,
                 name='online_', 
                 internal_dim = 256,
                 tsaills = False,
                 k=0.5, 
                 q=2.0,
                 tau=0.005,
                 sample_policy = 'softmax',
                 use_trust_region = False):
        # super(RNNPCLModel, self).__init__(env_spec)
        super(RNNPCLModel, self).__init__(self, env_spec)
        self.env_spec = env_spec
        self.input_time_step = input_time_step
        self.input_policy_state = input_policy_state
        #!!
        # self.input_prev_actions = False
        self.input_prev_actions = input_prev_actions
        self.fixed_std = fixed_std

        self.internal_dim = internal_dim
        self.rnn_cell_num = self.internal_dim // 2
        # self.rnn_cell_num = self.internal_dim 
        self.tsaills = tsaills

        self.k = k
        self.q = q
        self.tau = tau
        self.sample_policy = sample_policy

        self.policy_encoder = PolicyEncoder(
            self.env_spec, internal_dim=self.internal_dim, input_prev_actions=self.input_prev_actions, name=name)
        self.LSTM_layer = LSTM(
            self.rnn_cell_num, stateful=True, return_state=True, name=f'{name}LSTM_layer')
        self.LSTM_wrapper = Dense(self.output_dim, kernel_initializer=dev_initializers, name=f'{name}LSTM_wrapper')

        self.value_layer = Dense(self.rnn_cell_num, name=f'{name}value_layer')
        # self.value_layer_2 = Dense(
        #     1, activation=None, name=f'{name}value_wrapper')

        if self.tsaills:
            self.value_layer_2 = Dense(
                1, activation=None, use_bias=False, kernel_initializer=dev_initializers, name=f'{name}value_wrapper')
            self.value_output_layer_lambda = Dense(
                1, activation=None, use_bias=False, kernel_initializer=dev_initializers, name=f'{name}value_wrapper_lambda')
            self.value_output_layer_L = Dense(
                1, activation=None, use_bias=False, kernel_initializer=dev_initializers, name=f'{name}value_wrapper_L')
        else:
            self.value_layer_2 = Dense(
                1, activation=None, use_bias=False, kernel_initializer=dev_initializers, name=f'{name}value_wrapper')

    @property
    def output_dim(self):
        return self.env_spec.total_sampling_act_dim

    # sample_step()
    def call(self, obs, act, state_h, state_c,):
        x = self.policy_encoder(obs, act)
        # the input shape for LSTM:(batch_size, timesteps, input_dim)
        # the added dimension is timesteps
        x = tf.expand_dims(x, 1)
        x, state_h, state_c = self.LSTM_layer(
            x, initial_state=[state_h, state_c])
        x = self.LSTM_wrapper(x)

        return x, state_h, state_c

    def forward(self, states):
        v_x = self.value_layer(states)

        if self.tsaills:
            v_pcl = self.value_layer_2(v_x)
            v_lambda = self.value_output_layer_lambda(v_x)
            v_L = self.value_output_layer_L(v_x)
            value = tf.stack([v_pcl, v_lambda, v_L])
        else:
            value = self.value_layer_2(v_x)
        return value, v_x

    # input: self.last_obs, self.internal_state, self.last_act
    # output: next_internal_state, sampled_actions, logits, log_probs, entropy, self_kl
    def single_step(self, last_obs, last_act, internal_state, cur_act=None):
        [state_h, state_c] = internal_state

        # TODO 源码会将初始化的action(全为-1)放进网络？
        outputs, new_state_h, new_state_c = self.call(
            last_obs, last_act, state_h, state_c)
        sampled_actions, logits, log_probs, entropy, self_kl = self.process_net_outputs(
            outputs, cur_act)

        return [new_state_h, new_state_c], sampled_actions, logits, log_probs, entropy, self_kl

    def multi_step(self, all_obs, initial_state, all_actions):
        """Calculate log-probs and other calculations on batch of episodes."""
        # 模式参考_sampling_episode()
        batch_size = tf.shape(all_obs[0])[1]
        time_length = tf.shape(all_obs[0])[0]

        def refactor_log_prob(log_probs):
            log_probs = log_probs[:-1]
            log_probs = tf.stack(log_probs)
            log_probs = tf.transpose(a=log_probs, perm=[1, 0, 2])
            log_probs = tuple(log_probs)
            return log_probs

        def refactor_logit(logit):
            logit = tf.convert_to_tensor(logit)
            logit = tf.squeeze(logit, axis=1)
            # logit = tf.reshape(logit, [time_length, batch_size, -1])
            logit = tf.expand_dims(logit, 0)
            logit = tuple(logit)
            return logit

        def refactor_entropy(entropy):
            entropy = entropy[:-1]
            entropy = tf.stack(entropy)
            entropy = tf.squeeze(entropy, axis=1)
            entropy = [entropy]
            return entropy

        all_states, all_log_probs, all_logits = [], [], []
        all_states_t, values_t, all_log_probs_t, all_entropy = [], [], [], []

        internal_state = initial_state
        all_states.append(initial_state)

        shift_action = [tf.concat([act[1:], act[:1]], 0)
                        for act in all_actions]

        for prev_act, obs, act in zip(all_actions[0], all_obs[0], shift_action[0]):
            prev_act = [prev_act]
            obs = [obs]

            next_internal_state, _, logits, log_probs, entropy, \
                _ = self.single_step(obs, prev_act, internal_state, act)

            all_states.append(next_internal_state)
            all_log_probs.append(log_probs)
            all_logits.append(logits)
            all_entropy.append(entropy)
            internal_state = next_internal_state

        # TODO 是否要抹掉最后一个元素
        all_log_probs = refactor_log_prob(all_log_probs)
        all_logits = refactor_logit(all_logits)
        all_entropy = refactor_entropy(all_entropy)

        all_states = all_states[:-1]

        return all_states, all_log_probs, all_logits, all_entropy

    def reset(self):
        self.LSTM_layer.reset_states()


class CNNPCLModel(PCLModel):
    # def __init__(self, env_spec, name='online_',  input_prev_actions=True, value_hidden_layers=2):
    def __init__(self, env_spec,  input_time_step=False,
                 input_policy_state=True,
                 input_prev_actions=False,
                 fixed_std=False,
                 value_hidden_layers=2,
                 name='online_', 
                 internal_dim = 256,
                 tsaills = False,
                 k=0.5, 
                 q=2.0,
                 tau=0.005,
                 sample_policy = 'softmax',
                 use_trust_region = False):
        self.env_spec = env_spec
        self.input_time_step = input_time_step
        self.input_policy_state = input_policy_state
        self.input_prev_actions = input_prev_actions
        self.fixed_std = fixed_std
        self.internal_dim = internal_dim
        self.tsaills = tsaills

        self.value_hidden_layers = value_hidden_layers
        self.model_name = name

        self.k = k
        self.q = q
        self.tau = tau
        self.sample_policy = sample_policy


        super(PCLModel, self).__init__()
        self.policy_encoder = PolicyEncoder(
            self.env_spec, internal_dim=self.internal_dim, input_prev_actions=self.input_prev_actions, name=name)
        self.policy_layer_1 = Dense(
            self.internal_dim, activation='tanh', name=f'{name}policy_layer_1')
        self.policy_layer_2 = Dense(
            self.internal_dim, activation='tanh', name=f'{name}policy_layer_2')
        self.policy_output_wrapper = Dense(
            self.output_dim, activation=None, use_bias=False, name=f'{name}policy_output_wrapper')

        self.value_layers = {}
        for i in range(self.value_hidden_layers):
            self.value_layers[name+'value_layer_%d' % i] \
                = Dense(self.internal_dim, activation='tanh', use_bias=False, name=name+'value_layer_%d' % i)

        if self.tsaills:
            self.value_output_layer = Dense(
                1, activation=None, use_bias=False, name=f'{name}value_wrapper')
            self.value_output_layer_lambda = Dense(
                1, activation=None, use_bias=False, name=f'{name}value_wrapper_lambda')
            self.value_output_layer_L = Dense(
                1, activation=None, use_bias=False, name=f'{name}value_wrapper_L')
        else:
            self.value_output_layer = Dense(
                1, activation=None, use_bias=False, name=f'{name}value_wrapper')

        self.log_std = {}
        # self.log_std_0 = tf.Variable([0.0, 0.0], dtype=tf.float32, name=f'{name}_try')
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
            sampling_dim = self.env_spec.sampling_dim(act_dim, act_type)

            # TODO continue space, may need trainable variable for sampling from a distribution
            if self.fixed_std and self.env_spec.is_box(act_type):
                self.log_std[name+'std%d' % i] = tf.Variable(np.random.random((1, sampling_dim // 2)), \
                                                        dtype=tf.float32, name=f'{name}std{i}')


    # sample_step()
    def call(self, obs, act):
        # state_h, state_h = tf.zeros(state_h.shape), tf.zeros(state_h.shape)

        x = self.policy_encoder(obs, act)
        # TODO这个应该不需要
        # x = tf.expand_dims(x, 1)
        x = self.policy_layer_1(x)
        x = self.policy_layer_2(x)
        x = self.policy_output_wrapper(x)

        return x

    def forward(self, inputs):
        # TODO 在controller中统一value
        v_x = inputs
        for i in range(self.value_hidden_layers):
            v_x = self.value_layers[self.model_name +
                                    'value_layer_%d' % i](v_x)
        if self.tsaills:
            v_pcl = self.value_output_layer(v_x)
            v_lambda = self.value_output_layer_lambda(v_x)
            v_L = self.value_output_layer_L(v_x)
            value = tf.stack([v_pcl, v_lambda, v_L])
        else:
            value = self.value_output_layer(v_x)
        return value, v_x

    def single_step(self, last_obs, last_act, internal_state, cur_act=None):
        [state_h, state_c] = internal_state

        outputs = self.call(last_obs, last_act)

        sampled_actions, logits, log_probs, entropy, self_kl = self.process_net_outputs(
            outputs, cur_act)

        return [state_h, state_c], sampled_actions, logits, log_probs, entropy, self_kl

    def multi_step(self, all_obs, initial_state, all_actions):
        """Calculate log-probs and other calculations on batch of episodes."""
        # 模式参考_sampling_episode()
        # *****prev*****
        def refactor_log_prob(log_probs):
            log_probs = log_probs[0]
            log_probs = tf.reshape(log_probs, [time_length, batch_size])
            log_probs = log_probs[:-1, :]
            log_probs = tf.expand_dims(log_probs, 0)
            log_probs = tuple(log_probs)
            return log_probs

        def refactor_logit(logit):
            logit = logit[0]
            logit = tf.reshape(logit, [time_length, batch_size, -1])
            logit = tf.expand_dims(logit, 0)
            logit = tuple(logit)
            return logit

        all_states, all_log_probs = [], []

        batch_size = tf.shape(all_obs[0])[1]
        time_length = tf.shape(all_obs[0])[0]
        # first reshape inputs as a single batch
        reshaped_obs = []
        for obs, (obs_dim, obs_type) in zip(all_obs, self.env_spec.obs_dims_and_types):
            if self.env_spec.is_discrete(obs_type):
                reshaped_obs.append(tf.reshape(
                    obs, [time_length * batch_size]))
            elif self.env_spec.is_box(obs_type):
                reshaped_obs.append(tf.reshape(
                    obs, [time_length * batch_size, obs_dim]))

        reshaped_act = []
        reshaped_prev_act = []
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
            act = tf.concat([all_actions[i][1:], all_actions[i][0:1]], 0)
            prev_act = all_actions[i]
            if self.env_spec.is_discrete(act_type):
                reshaped_act.append(
                    tf.reshape(act,      [time_length * batch_size]))
                reshaped_prev_act.append(
                    tf.reshape(prev_act, [time_length * batch_size]))
            elif self.env_spec.is_box(act_type):
                reshaped_act.append(
                    tf.reshape(act, [time_length * batch_size, act_dim]))
                reshaped_prev_act.append(
                    tf.reshape(prev_act, [time_length * batch_size, act_dim]))

        (internal_states, _, logits, log_probs,
         entropies, self_kls) = self.single_step(
            reshaped_obs, reshaped_act, initial_state, reshaped_prev_act)

        # TODO 是否要抹掉最后一个元素
        all_log_probs = refactor_log_prob(log_probs)
        all_logits = refactor_logit(logits)


        all_states = tf.zeros([time_length, 2, batch_size, 128])

        return all_states, all_log_probs, all_logits, entropies

    def reset(self):
        return


if __name__ == "__main__":

    env_name = 'Copy-v0'
    env = gym_wrapper.Environment(env_name)
    env_spec_ = env_spec.EnvSpec(env.get_one())
    model = PCLModel(env_spec_)
    batch_size = 20
    obs_act_dim = 10  # 实际中obs和act的dim并不相同
    obs = np.random.random((batch_size, obs_act_dim,)).astype(np.float32)
    act = np.random.random((batch_size, obs_act_dim,)).astype(np.float32)
    obs_1 = np.random.random((batch_size, obs_act_dim,)).astype(np.float32)
    act_1 = np.random.random((batch_size, obs_act_dim,)).astype(np.float32)
    pre_all_states = np.random.random((batch_size, 128,)).astype(np.float32)

    # * test layer
    encoder = PolicyEncoder(env_spec_)
    # print(encoder.weights)
    output1 = encoder(obs, act)
    print(output1.numpy()[0, 0])
    output2 = encoder(obs, act)
    print(output2.numpy()[0, 0])
    # print(output)

    # * test mdoel
    # data1 = np.random.random((batch_size, 10, 50)).astype(np.float32)
    # data2 = np.random.random((batch_size, 10, 50)).astype(np.float32)
    pre_state_h, pre_state_c = tf.zeros(
        [obs_act_dim, 128]), tf.zeros([obs_act_dim, 128])
    # pre_states =np.random.random([64, 64]).astype(np.float32)
    _, state_h1, state_c1 = model(obs, act, pre_state_h, pre_state_c)
    outputs1, state_h2, state_c2 = model(obs_1, act_1, state_h1, state_c1)
    # model.LSTM_layer.reset_states((state_h1.numpy(), state_c1.numpy()))
    model.LSTM_layer.reset_states()
    # print(type(outputs1))
    # print(outputs1.numpy().shape)
    outputs1 = outputs1.numpy()[1, 1]

