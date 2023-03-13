import tensorflow as tf
import numpy as np

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

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.cur_step = 0
        self.writer = None
        self.validation_frequency = 0

    def discounted_future_sum(self, values, discount, rollout):
        """Discounted future sum of time-major values."""
        discount_filter = tf.reshape(
            discount ** tf.range(float(rollout)), [-1, 1, 1])
        expanded_values = tf.concat(
            [values, tf.zeros([rollout - 1, tf.shape(input=values)[1]])], 0)

        conv_values = tf.transpose(a=tf.squeeze(tf.nn.conv1d(
            input=tf.expand_dims(tf.transpose(a=expanded_values), -1), filters=discount_filter,
            stride=1, padding='VALID'), -1))

        return conv_values

    def shift_values(self, values, gamma, rollout, final_values=0.0):
        """Shift values up by some amount of time.

        Those values that shift from a value beyond the last value
        are calculated using final_values.

        """
        roll_range = tf.cumsum(tf.ones_like(values[:rollout, :]), 0, exclusive=True, reverse=True)

        final_pad = tf.expand_dims(final_values, 0) * gamma ** roll_range
        return tf.concat([gamma ** rollout * values[rollout:, :], final_pad], 0)
    
    def set_writer(self, writer, validation_frequency):
        self.writer = writer
        self.validation_frequency = validation_frequency

    def set_step(self, step):
        self.cur_step = step

    def write_data(self):
        return self.cur_step % self.validation_frequency == 0

class ActorCritic(Objective):
    """Standard Actor-Critic."""

    def __init__(self, learning_rate=0.01, clip_norm=5,
                policy_weight=1.0, critic_weight=1.0,
                tau=0.025, gamma=0.9, rollout=10,
                eps_lambda=0.0, clip_adv=None,
                use_target_values=False,
                max_divergence = 0):
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout

        self.clip_adv = clip_adv

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def grad(self, rewards, pads, values, final_values, log_probs, logits,\
         values_t_,final_values_t, all_log_probs_t, target_logits, entropies, actions=None):
        loss_value, raw_loss, reg_target = self.loss(rewards, pads, values, final_values, log_probs,\
                                               values_t_,final_values_t, all_log_probs_t, entropies)
        return loss_value, reg_target


    def loss(self, rewards, pads, values, final_values,
          log_probs,target_values, final_target_values, target_log_probs, entropies):

        pads = tf.convert_to_tensor(value=np.array(pads))
        pads = tf.cast(pads, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
          
        not_pad = 1 - pads
        batch_size = tf.shape(input=rewards)[1]
        a = sum(entropies)

        entropy = not_pad * sum(entropies)
        rewards = not_pad * rewards
        value_estimates = not_pad * values
        log_probs = not_pad * sum(log_probs)

        sum_rewards = self.discounted_future_sum(rewards, self.gamma, self.rollout)
        # if self.use_target_values:
        #     last_values = self.shift_values(
        #         target_values, self.gamma, self.rollout,
        #         final_target_values)
        # else:
        #     last_values = self.shift_values(value_estimates, self.gamma, self.rollout,
        #                                 final_values)
        last_values = self.shift_values(value_estimates, self.gamma, self.rollout,
                                    final_values)
        future_values = sum_rewards + last_values
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        if self.clip_adv:
            adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
        policy_loss = -adv * log_probs
        critic_loss = -adv * baseline_values
        regularizer = -self.tau * entropy

        policy_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=policy_loss * not_pad, axis=0))
        critic_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=critic_loss * not_pad, axis=0))
        regularizer = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=regularizer * not_pad, axis=0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss + regularizer)

        raw_loss = tf.reduce_mean(  # TODO
            input_tensor=tf.reduce_sum(input_tensor=not_pad * policy_loss, axis=0))

        if self.write_data():
            with self.writer.as_default():
                tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0), step=self.cur_step)
                tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0), step=self.cur_step)
                tf.summary.histogram('future_values', future_values, step=self.cur_step)
                tf.summary.histogram('baseline_values', baseline_values, step=self.cur_step)
                tf.summary.histogram('advantages', adv, step=self.cur_step)
                tf.summary.scalar('avg_rewards',
                                tf.reduce_mean(tf.reduce_sum(rewards, 0)), step=self.cur_step)
                tf.summary.scalar('policy_loss',
                                tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)), step=self.cur_step)
                tf.summary.scalar('critic_loss',
                                tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)), step=self.cur_step)
                tf.summary.scalar('loss', loss, step=self.cur_step)
                tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss), step=self.cur_step)
                # tf.summary.scalar('eps_lambda', self.eps_lambda, step=self.cur_step)

        return (loss, raw_loss,
                future_values[self.rollout - 1:, :])


class PCL(Objective):
    def __init__(self, learning_rate=0.01, clip_norm=5,
                policy_weight=1.0, critic_weight=1.0,
                tau=0.025, gamma=0.9, rollout=10,
                eps_lambda=0.0, clip_adv=None,
                use_target_values=False,
                max_divergence = 0):
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def grad(self, rewards, pads, values, final_values, log_probs, logits,\
         values_t_,final_values_t, all_log_probs_t, target_logits, entropies, actions=None):
        loss_value, raw_loss, reg_target = self.loss(rewards, pads, values, final_values, log_probs)
        return loss_value, reg_target

    #TODO trace back the dimensions of the variable
    def loss(self, rewards, pads, values, final_values, log_probs):
        pads = tf.convert_to_tensor(value=np.array(pads))
        pads = tf.cast(pads, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        
        not_pad = 1 - pads
        batch_size = tf.shape(input=rewards)[1]
        rewards            = not_pad * rewards
        value_estimates    = not_pad * values
        log_probs          = not_pad * sum(log_probs)

        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                            not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                            rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
            tf.ones([self.rollout - 1, batch_size]) *
            value_estimates[0:1, :],
            value_estimates], 0)
        log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                            log_probs], 0)

        sum_rewards            = self.discounted_future_sum(rewards, self.gamma, self.rollout)
        sum_log_probs          = self.discounted_future_sum(log_probs, self.gamma, self.rollout)

        roll_range_ = tf.cumsum(tf.ones_like(value_estimates[:self.rollout, :]), 0,
                                exclusive=True, reverse=True)
        final_pad_ = tf.expand_dims(final_values, 0) * self.gamma ** roll_range_
        last_values = tf.concat([self.gamma ** self.rollout * value_estimates[self.rollout:, :],
                            final_pad_], 0)
                                    
        future_values = (
            - self.tau * sum_log_probs
            # - self.eps_lambda * sum_relative_log_probs
            + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        # if self.clip_adv:
        #     adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
        policy_loss = -adv * sum_log_probs
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=policy_loss * not_pad, axis=0))
        critic_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=critic_loss * not_pad, axis=0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=not_pad * adv * (-baseline_values + future_values), axis=0))

        return (loss, raw_loss,
                future_values[self.rollout - 1:, :])


class TrustPCL(Objective):
    def __init__(self, learning_rate=0.01, clip_norm=5,
                policy_weight=1.0, critic_weight=1.0,
                tau=0.025, gamma=0.9, rollout=10,
                eps_lambda=0.0, clip_adv=None,
                use_target_values=False, max_divergence = 0.5):
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout
        self.use_target_values = use_target_values

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=2e-4)
        self.max_divergence = max_divergence 
        self.eps_lambda = eps_lambda

        self.clip_adv = 1

    def grad(self, rewards, pads, values, final_values, log_probs, logits,\
         values_t_,final_values_t, all_log_probs_t, target_logits, entropies, actions=None):
        loss_value, raw_loss, reg_target = self.loss(rewards, pads, values, final_values, log_probs,\
                                               values_t_,final_values_t, all_log_probs_t)
        return loss_value, reg_target

    def loss(self, rewards, pads, values, final_values,
          log_probs,target_values, final_target_values, target_log_probs):
          
        pads = tf.convert_to_tensor(value=np.array(pads))
        pads = tf.cast(pads, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        
        not_pad = 1 - pads
        batch_size = tf.shape(input=rewards)[1]
        sum_log_probs1 = sum(log_probs)


        rewards            = not_pad * rewards
        value_estimates    = not_pad * values
        log_probs          = not_pad * sum(log_probs)
        
        target_log_probs   = not_pad * tf.stop_gradient(sum(target_log_probs))
        relative_log_probs = not_pad * (log_probs - target_log_probs)
        target_values      = not_pad * tf.stop_gradient(target_values)
        final_target_values = tf.stop_gradient(final_target_values)

        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                            not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                            rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
            tf.ones([self.rollout - 1, batch_size]) *
            value_estimates[0:1, :],
            value_estimates], 0)
        log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                            log_probs], 0)
        # prev_log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
        #                             prev_log_probs], 0)

        relative_log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                                        relative_log_probs], 0)
        target_values = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
            tf.ones([self.rollout - 1, batch_size]) *
            target_values[0:1, :],
            target_values], 0)

        sum_rewards            = self.discounted_future_sum(rewards, self.gamma, self.rollout)
        sum_log_probs          = self.discounted_future_sum(log_probs, self.gamma, self.rollout)
        # sum_prev_log_probs     = self.discounted_future_sum(prev_log_probs, self.gamma, self.rollout)
        sum_relative_log_probs = self.discounted_future_sum(
            relative_log_probs, self.gamma, self.rollout)

        if self.use_target_values:
            last_values = self.shift_values(
                target_values, self.gamma, self.rollout,
                final_target_values)
        else:
            last_values = self.shift_values(value_estimates, self.gamma, self.rollout,
                                        final_values)
        future_values = (
            - self.tau * sum_log_probs
            - self.eps_lambda * sum_relative_log_probs
            + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        if self.clip_adv:
            adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
        policy_loss = -adv * sum_log_probs
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=policy_loss * not_pad, axis=0))
        critic_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=critic_loss * not_pad, axis=0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=not_pad * adv * (-baseline_values + future_values), axis=0))

        if self.write_data():
            with self.writer.as_default():
                tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0), step=self.cur_step)
                tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0), step=self.cur_step)
                tf.summary.histogram('future_values', future_values, step=self.cur_step)
                tf.summary.histogram('baseline_values', baseline_values, step=self.cur_step)
                tf.summary.histogram('advantages', adv, step=self.cur_step)
                tf.summary.scalar('avg_rewards',
                                tf.reduce_mean(tf.reduce_sum(rewards, 0)), step=self.cur_step)
                tf.summary.scalar('policy_loss',
                                tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)), step=self.cur_step)
                tf.summary.scalar('critic_loss',
                                tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)), step=self.cur_step)
                tf.summary.scalar('loss', loss, step=self.cur_step)
                tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss), step=self.cur_step)
                tf.summary.scalar('eps_lambda', self.eps_lambda, step=self.cur_step)
        
        return (loss, raw_loss,
                future_values[self.rollout - 1:, :])

    def update_lambda(self, new_lambda):
        self.eps_lambda = 0.01 * new_lambda + 0.99 * self.eps_lambda
        # print('eps_lambda: ', self.eps_lambda)

    def discounted_future_sum(self, values, discount, rollout):
        """Discounted future sum of time-major values."""
        discount_filter = tf.reshape(
            discount ** tf.range(float(rollout)), [-1, 1, 1])
        expanded_values = tf.concat(
            [values, tf.zeros([rollout - 1, tf.shape(input=values)[1]])], 0)

        conv_values = tf.transpose(a=tf.squeeze(tf.nn.conv1d(
            input=tf.expand_dims(tf.transpose(a=expanded_values), -1), filters=discount_filter,
            stride=1, padding='VALID'), -1))

        return conv_values


class SparsePCL(Objective):
    def __init__(self, learning_rate=0.01, clip_norm=5,
                policy_weight=1.0, critic_weight=1.0,
                tau=0.025, gamma=0.9, rollout=10,
                eps_lambda=0.0, clip_adv=None,
                use_target_values=False, max_divergence = 0.5):
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout
        self.use_target_values = use_target_values

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=2e-4)
        self.max_divergence = max_divergence 
        self.eps_lambda = eps_lambda

        self.clip_adv = 1

    def grad(self, rewards, pads, values, final_values, log_probs, logits,\
         values_t_,final_values_t, all_log_probs_t, target_logits, entropies, actions=None):

        loss_value, raw_loss, reg_target = self.loss(rewards, pads, values, final_values, log_probs, logits,\
                                               values_t_,final_values_t, all_log_probs_t, target_logits, actions)
        return loss_value, reg_target

    def loss(self, rewards, pads, values, final_values, log_probs, logits, 
            target_values, final_target_values, target_log_probs, target_logits, actions=None):
        assert len(logits) == 1, 'only one discrete action allowed'
        assert actions is not None

        pads = tf.convert_to_tensor(value=np.array(pads))
        pads = tf.cast(pads, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        actions = tf.convert_to_tensor(value=actions)
        actions = tf.squeeze(actions, axis=0)[1:, :]

        not_pad = 1 - pads
        time_length = tf.shape(input=rewards)[0]
        batch_size = tf.shape(input=rewards)[1]
        num_actions = tf.shape(input=logits[0])[2]

        rewards = not_pad * rewards
        value_estimates = not_pad * values[:, :, 0]
        lambda_coefs = tf.exp(1.0 * not_pad * values[:, :, 1])
        Lambda_sigmoid = not_pad * tf.sigmoid(values[:, :, 2])

        #remove final computation

        logits = logits[0][:-1]
        # logits = [logit[:-1] for logit in logits] #logits[:-1]  # [:-1]

        tau_logits = tf.reshape(
            spmax_tau(tf.reshape(logits, [time_length * batch_size, -1])),
            [time_length, batch_size, 1])

        pi_probs = not_pad * tf.reduce_sum(
            input_tensor=tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        lambdas = not_pad * tf.reduce_sum(
            input_tensor=tf.nn.relu(tau_logits - logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        Lambdas = Lambda_sigmoid * (-self.tau / 2)

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        lambda_coefs = tf.concat(
            [tf.ones([self.rollout - 1, batch_size]),
             lambda_coefs], 0)

        pi_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs], 0)
        lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             lambdas], 0)
        Lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             Lambdas], 0)

        sum_rewards = self.discounted_future_sum(rewards + self.tau / 2, self.gamma, self.rollout)
        sum_pi_probs = self.discounted_future_sum(pi_probs, self.gamma, self.rollout)
        sum_lambdas = self.discounted_future_sum(lambdas * lambda_coefs, self.gamma, self.rollout)
        sum_Lambdas = self.discounted_future_sum(Lambdas, self.gamma, self.rollout)

        # last_values = tf.stop_gradient(shift_values(value_estimates, self.gamma, self.rollout))
        last_values = self.shift_values(value_estimates, self.gamma, self.rollout)

        future_values = (
                - self.tau * sum_pi_probs
                + self.tau * sum_lambdas
                # + sum_lambdas
                - sum_Lambdas
                + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        if self.clip_adv:
            adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))

        policy_loss = -adv * (sum_pi_probs - sum_lambdas + sum_Lambdas)
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=policy_loss * not_pad, axis=0))
        critic_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=critic_loss * not_pad, axis=0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=not_pad * adv * (-baseline_values + future_values), axis=0))

        # gradient_ops = self.training_ops(
        #     loss, learning_rate=self.learning_rate)

        if self.write_data():
            with self.writer.as_default():
                tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0), step=self.cur_step)
                tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0), step=self.cur_step)
                tf.summary.histogram('future_values', future_values, step=self.cur_step)
                tf.summary.histogram('baseline_values', baseline_values, step=self.cur_step)
                tf.summary.histogram('advantages', adv, step=self.cur_step)
                tf.summary.scalar('avg_rewards',
                                tf.reduce_mean(tf.reduce_sum(rewards, 0)), step=self.cur_step)
                tf.summary.scalar('policy_loss',
                                tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)), step=self.cur_step)
                tf.summary.scalar('critic_loss',
                                tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)), step=self.cur_step)
                tf.summary.scalar('loss', loss, step=self.cur_step)
                tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss), step=self.cur_step)
                tf.summary.scalar('eps_lambda', self.eps_lambda, step=self.cur_step)

        return (loss, raw_loss,
                future_values[self.rollout - 1:, :])


class GeneralSparsePCL(Objective):

    def get(self, rewards, pads, values, final_values,
            log_probs, prev_log_probs, target_log_probs,
            entropies, logits,
            target_values, final_target_values, actions=None):
        assert len(logits) == 1, 'only one discrete action allowed'
        assert actions is not None

        not_pad = 1 - pads
        time_length = tf.shape(input=rewards)[0]
        batch_size = tf.shape(input=rewards)[1]
        num_actions = tf.shape(input=logits[0])[2]

        rewards = not_pad * rewards
        value_estimates = not_pad * values[:, :, 0]
        lambda_coefs = tf.exp(1.0 * not_pad * values[:, :, 1])
        Lambda_sigmoid = not_pad * tf.sigmoid(values[:, :, 2])

        #remove final computation

        logits = logits[0][:-1]#/(self.k * self.q)
        # logits = logits[0][:-1]

        # logits = [logit[:-1] for logit in logits] #logits[:-1]  # [:-1]

        tau_logits = tf.reshape(
            spmax_tau(tf.reshape(logits, [time_length * batch_size, -1])),
            [time_length, batch_size, 1])

        pi_probs = not_pad * tf.reduce_sum(
            input_tensor=tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        pi_probs = tf.pow(pi_probs, (self.q - 1))

        lambdas = not_pad * tf.reduce_sum(
            input_tensor=tf.nn.relu(tau_logits - logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        Lambdas = Lambda_sigmoid * (-self.tau * self.k/(self.q - 1))

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        lambda_coefs = tf.concat(
            [tf.ones([self.rollout - 1, batch_size]),
             lambda_coefs], 0)
        pi_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs], 0)
        lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             lambdas], 0)
        Lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             Lambdas], 0)

        sum_rewards = self.discounted_future_sum(rewards + self.k * self.tau / (self.q - 1), self.gamma, self.rollout)
        sum_pi_probs = self.discounted_future_sum(pi_probs, self.gamma, self.rollout)
        sum_lambdas = self.discounted_future_sum(lambdas * lambda_coefs, self.gamma, self.rollout)
        sum_Lambdas = self.discounted_future_sum(Lambdas, self.gamma, self.rollout)

        # last_values = tf.stop_gradient(shift_values(value_estimates, self.gamma, self.rollout))
        last_values = self.shift_values(value_estimates, self.gamma, self.rollout)

        future_values = (
                - ((self.tau * self.q * self.k) / (self.q - 1)) * sum_pi_probs
                # + self.tau * sum_lambdas
                + sum_lambdas
                - sum_Lambdas
                + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        raw_adv = adv

        policy_loss = -adv * (sum_pi_probs - sum_lambdas + sum_Lambdas)
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=policy_loss * not_pad, axis=0))
        critic_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=critic_loss * not_pad, axis=0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=not_pad * adv * (-baseline_values + future_values), axis=0))

        gradient_ops = self.training_ops(
            loss, learning_rate=self.learning_rate)

        tf.compat.v1.summary.histogram('log_probs', tf.reduce_sum(input_tensor=log_probs, axis=0))
        tf.compat.v1.summary.histogram('rewards', tf.reduce_sum(input_tensor=rewards, axis=0))
        tf.compat.v1.summary.histogram('future_values', future_values)
        tf.compat.v1.summary.histogram('baseline_values', baseline_values)
        tf.compat.v1.summary.histogram('advantages', adv)
        tf.compat.v1.summary.scalar('avg_rewards',
                          tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=rewards, axis=0)))
        tf.compat.v1.summary.scalar('policy_loss',
                          tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=not_pad * policy_loss)))
        tf.compat.v1.summary.scalar('critic_loss',
                          tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=not_pad * policy_loss)))
        tf.compat.v1.summary.scalar('loss', loss)
        tf.compat.v1.summary.scalar('raw_loss', tf.reduce_mean(input_tensor=raw_loss))
        tf.compat.v1.summary.scalar('eps_lambda', self.eps_lambda)

        return (loss, raw_loss, future_values,
                gradient_ops, tf.compat.v1.summary.merge_all())

class GeneralSparsePCLV2(Objective):


    def get(self, rewards, pads, values, final_values,
            log_probs, prev_log_probs, target_log_probs,
            entropies, logits,
            target_values, final_target_values, actions=None):
        assert len(logits) == 1, 'only one discrete action allowed'
        assert actions is not None

        not_pad = 1 - pads
        time_length = tf.shape(input=rewards)[0]
        batch_size = tf.shape(input=rewards)[1]
        num_actions = tf.shape(input=logits[0])[2]

        rewards = not_pad * rewards
        value_estimates = not_pad * values[:, :, 0]
        lambda_coefs = tf.exp(1.0 * not_pad * values[:, :, 1])
        Lambda_sigmoid = not_pad * tf.sigmoid(values[:, :, 2])

        #remove final computation

        # big_o = 0.00001 * (self.q - 2)
        big_o = 1

        logits = logits[0][:-1]#/(self.k * self.q)
        # logits = [logit[:-1] for logit in logits] #logits[:-1]  # [:-1]

        tau_logits = tf.reshape(
            spmax_tau(tf.reshape(logits, [time_length * batch_size, -1])),
            [time_length, batch_size, 1])

        pi_probs_1 = not_pad * tf.reduce_sum(
            # tf.nn.relu(logits - tau_logits + big_o/((self.q-1)*(self.q-1))) * tf.one_hot(actions, num_actions),
            input_tensor=tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        pi_probs_2 = not_pad * tf.reduce_sum(
            # tf.nn.relu(logits - tau_logits + big_o/((self.q-1)*(self.q-1))) * tf.one_hot(actions, num_actions),
            input_tensor=tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        lambdas = not_pad * tf.reduce_sum(
            # tf.nn.relu(tau_logits - logits - big_o/((self.q-1)*(self.q-1))) * tf.one_hot(actions, num_actions),
            input_tensor=tf.nn.relu(tau_logits - logits) * tf.one_hot(actions, num_actions),
            axis=-1)

        Lambdas = Lambda_sigmoid * (-2 * self.tau * self.k/(self.q + 1))

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        lambda_coefs = tf.concat(
            [tf.ones([self.rollout - 1, batch_size]),
             lambda_coefs], 0)
        pi_probs_1 = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs_1], 0)

        pi_probs_1 = tf.pow(pi_probs_1, (self.q - 1))

        pi_probs_2 = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs_2], 0)

        pi_probs_2 = tf.pow(pi_probs_2, (self.q + 1))


        lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             lambdas], 0)
        Lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             Lambdas], 0)

        sum_rewards = self.discounted_future_sum(rewards + self.k * self.tau / (self.q - 1), self.gamma, self.rollout)
        sum_pi_probs_1 = self.discounted_future_sum(pi_probs_1, self.gamma, self.rollout)
        sum_pi_probs_2 = self.discounted_future_sum(pi_probs_2, self.gamma, self.rollout)
        sum_lambdas = self.discounted_future_sum(lambdas * lambda_coefs, self.gamma, self.rollout)
        sum_Lambdas = self.discounted_future_sum(Lambdas, self.gamma, self.rollout)

        # last_values = tf.stop_gradient(shift_values(value_estimates, self.gamma, self.rollout))
        last_values = self.shift_values(value_estimates, self.gamma, self.rollout)

        future_values = (
                - ((self.tau * (self.k * self.q)) / (self.q - 1)) * sum_pi_probs_1
                + ((self.tau * self.k * (self.q-1)) / (self.q + 1)) * sum_pi_probs_2
                # + (self.tau * self.k) * sum_lambdas
                + sum_lambdas
                - sum_Lambdas
                + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        raw_adv = adv

        policy_loss = -adv * (sum_pi_probs_1 - sum_pi_probs_2 - sum_lambdas + sum_Lambdas)
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=policy_loss * not_pad, axis=0))
        critic_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=critic_loss * not_pad, axis=0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=not_pad * adv * (-baseline_values + future_values), axis=0))

        gradient_ops = self.training_ops(
            loss, learning_rate=self.learning_rate)

        tf.compat.v1.summary.histogram('log_probs', tf.reduce_sum(input_tensor=log_probs, axis=0))
        tf.compat.v1.summary.histogram('rewards', tf.reduce_sum(input_tensor=rewards, axis=0))
        tf.compat.v1.summary.histogram('future_values', future_values)
        tf.compat.v1.summary.histogram('baseline_values', baseline_values)
        tf.compat.v1.summary.histogram('advantages', adv)
        tf.compat.v1.summary.scalar('avg_rewards',
                          tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=rewards, axis=0)))
        tf.compat.v1.summary.scalar('policy_loss',
                          tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=not_pad * policy_loss)))
        tf.compat.v1.summary.scalar('critic_loss',
                          tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=not_pad * policy_loss)))
        tf.compat.v1.summary.scalar('loss', loss)
        tf.compat.v1.summary.scalar('raw_loss', tf.reduce_mean(input_tensor=raw_loss))
        tf.compat.v1.summary.scalar('eps_lambda', self.eps_lambda)

        return (loss, raw_loss, future_values,
                gradient_ops, tf.compat.v1.summary.merge_all())

