from typing import Iterator
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_layers import Dense, LSTM, NewPolicyEncoder

class NewCritic(nn.Module):
    def __init__(self, env_spec, policy_cell_num, name='online_'):
        #! deleted 
        super(NewCritic, self).__init__()
        self.env_spec = env_spec

        self.policy_cell_num = policy_cell_num
        self.tmp_infeature = 61
        #TODO the attr here

        self.input_time_step = True
        self.input_policy_state = True
        self.input_prev_actions = False

        self.internal_dim = self.set_internal_dim()        
        self.CreateCriticNetwork()

    def CreateCriticNetwork(self):
        self.value_layer = Dense(in_features=self.internal_dim, out_features=self.tmp_infeature)
        self.value_layer_2 = Dense(in_features=self.tmp_infeature,out_features=1)

    
    def set_internal_dim(self):
        """Get inputs to network as single tensor."""
        input_dim = 1

        if not self.input_policy_state:
            for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
                if self.env_spec.is_discrete(obs_type):
                    input_dim += obs_dim
                elif self.env_spec.is_box(obs_type):
                    input_dim += obs_dim * 2
                else:
                    assert False

            # LSTM:self.input_prev_actions = False
            if self.input_prev_actions:
                for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
                    if self.env_spec.is_discrete(act_type):
                        input_dim += act_dim
                    elif self.env_spec.is_box(act_type):
                        input_dim += act_dim
                    else:
                        assert False

        if self.input_policy_state:
            input_dim += self.policy_cell_num * 2


        # # LSTM:self.input_time_step = False
        if self.input_time_step:
            input_dim += 3  

        return input_dim        

    @property
    def output_dim(self):
        return self.env_spec.total_sampling_act_dim
    
    def forward(self, states):
        v_x = self.value_layer(states)
        value = self.value_layer_2(v_x)
        return value, v_x

    def get_values(self, all_obs, all_actions, all_states):
        """_summary_

        Args:
            all_obs (List(Tensor(time_length+1, bs))): _description_
            all_actions (List(Tensor(time_length+1, bs))): _description_
            all_states (List(Tensor(time_length+1, bs, 2, internal_dim))): _description_

        Returns:
            Tensor(time_length+1, bs, 2, internal_dim): values
        """
        batch_size = all_obs[0].shape[1]
        time_length = all_obs[0].shape[0]

        (time_step, reshaped_obs, reshaped_prev_act,
         reshaped_internal_policy_states) = self._reshape_batched_inputs(
            all_obs, all_actions, all_states)

        input_dim, inputs = self._get_inputs(
            time_step, reshaped_obs, reshaped_prev_act,
            reshaped_internal_policy_states)

        values, _ = self.forward(inputs)

        values = torch.reshape(values, [time_length, batch_size])

        return values

    def _reshape_batched_inputs(self, all_obs, all_actions,
                               internal_policy_states):
        """Reshape inputs from [time_length, batch_size, ...] to
        [time_length * batch_size, ...].

        This allows for computing the value estimate in one go.
        """
        batch_size = all_obs[0].shape[1]
        time_length = all_obs[0].shape[0]
        internal_policy_dim = internal_policy_states.shape[3]
        reshaped_internal_policy_dim = internal_policy_dim * 2

        reshaped_obs = []
        # # self.env_spec.obs_dims_and_types=((6,0),) len=100
        # # obs_dim=6, obs_type=0
        for obs, (obs_dim, obs_type) in zip(all_obs, self.env_spec.obs_dims_and_types):
            if self.env_spec.is_discrete(obs_type):
                reshaped_obs.append(torch.reshape(obs, [time_length * batch_size]))
                    
            elif self.env_spec.is_box(obs_type):
                reshaped_obs.append(torch.reshape(
                    obs, [time_length * batch_size, obs_dim]))

        reshaped_prev_act = []
        # # self.env_spec.act_dims_and_types=((20,0),)
        # # act_dim=20, act_type=0
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
            prev_act = all_actions[i]
            if self.env_spec.is_discrete(act_type):
                reshaped_prev_act.append(
                    torch.reshape(prev_act, [time_length * batch_size]))
            elif self.env_spec.is_box(act_type):
                reshaped_prev_act.append(
                    torch.reshape(prev_act, [time_length * batch_size, act_dim]))
        reshaped_internal_policy_states = torch.reshape(
            internal_policy_states,
            [time_length * batch_size, reshaped_internal_policy_dim])


        time_step = (float(self.input_time_step) * torch.unsqueeze(
            (torch.arange(start=0, end=time_length * batch_size) / batch_size).to(torch.float32), -1))
                    

        return (time_step, reshaped_obs, reshaped_prev_act,
                reshaped_internal_policy_states)

    def _get_inputs(self, time_step, obs, prev_actions,
                   internal_policy_states):
        """Get inputs to network as single tensor."""
        inputs = [torch.ones_like(time_step)]
        input_dim = 1

        if not self.input_policy_state:
            for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
                if self.env_spec.is_discrete(obs_type):
                    inputs.append(
                        torch.nn.functional.one_hot(obs[i], obs_dim))
                    input_dim += obs_dim
                elif self.env_spec.is_box(obs_type):
                    cur_obs = obs[i].to(torch.float32)
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
                            torch.nn.functional.one_hot(prev_actions[i], act_dim))
                        input_dim += act_dim
                    elif self.env_spec.is_box(act_type):
                        inputs.append(prev_actions[i])
                        input_dim += act_dim
                    else:
                        assert False

        if self.input_policy_state:
            inputs.append(internal_policy_states)
            input_dim += internal_policy_states.shape[1]

        # # LSTM:self.input_time_step = False
        if self.input_time_step:
            scaled_time = 0.01 * time_step
            inputs.extend([scaled_time, scaled_time ** 2, scaled_time ** 3])
            input_dim += 3

        return input_dim, torch.cat(inputs, 1)




