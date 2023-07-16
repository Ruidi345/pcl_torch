import torch
import torch.nn as nn
from torch_layers import Dense, LSTM, NewPolicyEncoder, spmax_tau
import numpy as np


class NewPolicy(nn.Module):
    def __init__(self, env_spec, internal_dim, input_prev_actions, lstm_hidden_size, name='online_', test=False):
        super(NewPolicy, self).__init__()
        self.env_spec = env_spec
        self.internal_dim = internal_dim
        self.FLAG_input_prev_actions = input_prev_actions
        self.test = test

        num_layers = 1

        # LSTM
        self.hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.lstm_input_size = self.output_dim

        self.CreatePolicyNetwork()

        self.sample_policy = 'softmax'
        #TODO wrap the follwoing in self.model

    def CreatePolicyNetwork(self):
        self.policy_encoder = NewPolicyEncoder(self.env_spec, internal_dim=self.lstm_input_size,
                                                input_prev_actions=self.FLAG_input_prev_actions)
        self.LSTM_layer = LSTM(input_size=self.lstm_input_size, output_size=self.hidden_size, num_layers=self.num_layers)
        self.LSTM_wrapper = Dense(in_features=self.hidden_size, out_features=self.output_dim)

    def sample_step(self, last_obs, last_act, internal_state):
        """_summary_

        Args:
            last_obs (list(Tensor(bs,), ..., Tensor(bs,))): _description_
            last_act (list(Tensor(bs,), ..., Tensor(bs,))): _description_
            internal_state (list(Tensor(1, bs, internal_dim,), Tensor(1, bs, internal_dim,))): _description_

        Returns:
            list(Tensor(1, bs, internal_dim,), Tensor(1, bs, internal_dim,)): inner state for LSTM
            list(Tensor(bs)): sampled action in this frame
            list(Tensor(bs)): log(pi(a | s)) in this frame
        """
        (state_h, state_c) = internal_state
  
        outputs, (new_state_h, new_state_c) = self.forward(last_obs, last_act, state_h, state_c)
        sampled_actions, log_probs = self._newsample_action(outputs)
            
        return [new_state_h, new_state_c], sampled_actions, log_probs
        
    @property
    def output_dim(self):
        return self.env_spec.total_sampling_act_dim
    
    def forward(self, last_obs, last_act, state_c, state_h):
        x = self.policy_encoder(last_obs, last_act)
        x = torch.unsqueeze(x, 0)
        # x, state_h, state_c = self.LSTM_layer(x, initial_state=[state_h, state_c])
        x, state = self.LSTM_layer(x, state_h, state_c)
        x = torch.squeeze(x, 0)         
        x = self.LSTM_wrapper(x) 
        return x, state
    

    def _newsample_action(self, output):
        """Sample all actions in a batch given output of core network."""
        sampled_actions = []
        log_probs = []

        start_idx = 0

        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
            sampling_dim = self.env_spec.sampling_dim(act_dim, act_type)
            # TODO continue space, may need trainable variable for sampling from a distribution
            act_logits = output[:, start_idx:start_idx + sampling_dim]

            act = self._sample_action(act_logits, sampling_dim, act_dim, act_type)
            act_log_prob = self._log_prob_action(
                act, act_logits, sampling_dim, act_dim, act_type)
            
            sampled_actions.append(act)            
            log_probs.append(act_log_prob)

            start_idx += sampling_dim

        assert start_idx == self.env_spec.total_sampling_act_dim

        return sampled_actions, log_probs

    def _sample_action(self, logits, sampling_dim,
                      act_dim, act_type, greedy=False):
        # sourcery skip: merge-duplicate-blocks
        """Sample an action from a distribution."""
        if self.env_spec.is_discrete(act_type):
            if self.sample_policy == 'greedy':
                act =torch.argmax(input=logits, axis=1)
            elif self.sample_policy == 'softmax':
                if self.test:
                    logits = torch.rand(logits.size())
                sampled = torch.multinomial(input=logits, num_samples=1)
                act = torch.reshape(sampled, [-1])
                    
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
                batch_size = torch.shape(logits)[0]
                act = means + std * torch.randn([batch_size, act_dim])
        else:
            assert False

        return act

    def _log_prob_action(self, action, logits,
                        sampling_dim, act_dim, act_type):
        # sourcery skip: merge-duplicate-blocks
        """Calculate log-prob of action sampled from distribution."""
        # action, len=100
        if self.env_spec.is_discrete(act_type):
            action = action.to(torch.int64)
            onehot_act = torch.nn.functional.one_hot(action, act_dim)
            logsm_logit = torch.nn.functional.log_softmax(logits)
            act_log_prob = torch.sum(onehot_act * logsm_logit, axis=-1)
                
        elif self.env_spec.is_box(act_type):
            # assert False
            div = int(sampling_dim/2)
            means = logits[:, :div]
            std = logits[:, div:]
            act_log_prob = (- 0.5 * torch.log(2 * np.pi * torch.square(std))
                            - 0.5 * torch.square(action - means) / torch.square(std))
            act_log_prob = torch.sum(act_log_prob, -1)
        else:
            assert False

        return act_log_prob

