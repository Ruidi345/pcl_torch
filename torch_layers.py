import torch
import torch.nn as nn


def spmax_tau(logits):
    batch_size = logits.size(0)
    num_actions = logits.size(1)

    z = logits

    z_sorted, _ = torch.sort(z, descending=True)

    z_cumsum = torch.cumsum(z_sorted, dim=1)
    k = torch.arange(1, num_actions + 1, dtype=logits.dtype, device=logits.device)
    z_check = 1 + k * z_sorted > z_cumsum

    k_z = torch.sum(z_check.int(), dim=1)

    indices = torch.stack([torch.arange(0, batch_size, dtype=torch.long, device=logits.device), k_z - 1], dim=1)
    tau_sum = torch.gather(z_cumsum, 1, indices)
    tau_z = (tau_sum - 1) / k_z.type(logits.dtype)

    return tau_z

class Dense(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False, init=False, activation=None, name=''):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.activation = activation
        if init:
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

        if name is not None:
            setattr(self, name, self.linear)

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, batch_first=False):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, output_size, num_layers, batch_first=batch_first)

    def forward(self, x, h, c):
        # h_state = (num_layers*num_directions, batch, output_size)
        # h0 = torch.zeros(self.num_layers, x.size(1), self.output_size)
        # c0 = torch.zeros(self.num_layers, x.size(1), self.output_size)
        # out, state = self.lstm(x, (h, c))
        # h.shape = (1, batch_size, internal_size)
        out, state = self.lstm(x, (h, c))
        return out, state

class NewPolicyEncoder(nn.Module):
    def __init__(self, env_spec, internal_dim, input_prev_actions=True, name='online') -> None:
        super(NewPolicyEncoder, self).__init__()
        self.env_spec = env_spec
        self.input_dim = internal_dim
        self.model_name = name
        self.input_prev_actions = input_prev_actions

        self.layer_vars = {}
        self.bias = torch.nn.Parameter(torch.empty(self.input_dim,))
            
        nn.init.constant_(self.bias, val=0)
        setattr(self, 'input_bias', self.bias)
        
        for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
            self.layer_vars[name+'w_state%d' % i] = torch.nn.Parameter(torch.empty(obs_dim, self.input_dim,))
            nn.init.normal_(self.layer_vars[name+'w_state%d' % i], mean=0.0, std=0.01)
            setattr(self, name+'w_state%d' % i, self.layer_vars[name+'w_state%d' % i])

        if self.input_prev_actions:
            if self.env_spec.combine_actions:
                for i, action_dim in enumerate(self.env_spec.orig_act_dims):
                    self.layer_vars[name+'w_prev_action%d' % i] = torch.nn.Parameter(torch.empty(action_dim, self.input_dim,))
                    nn.init.normal_(self.layer_vars[name+'w_prev_action%d' % i], mean=0.0, std=0.01)
                    setattr(self, name+'w_prev_action%d' % i, self.layer_vars[name+'w_prev_action%d' % i])
            else:
                for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
                    self.layer_vars[name+'w_prev_action%d' % i] = torch.nn.Parameter(torch.empty(act_dim, self.input_dim,))
                    nn.init.normal_(self.layer_vars[name+'w_prev_action%d' % i], mean=0.0, std=0.01)
                    setattr(self, name+'w_prev_action%d' % i, self.layer_vars[name+'w_prev_action%d' % i])

                    
    def forward(self, observations, prev_actions):
        batch_size = prev_actions[0].shape[0]
        #? self.input_dim = internal_dim
        cell_input = torch.empty(batch_size, self.input_dim)
        cell_input_expand = torch.unsqueeze(cell_input, 0)

        # reshape observation
        for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
            if self.env_spec.is_discrete(obs_type):
                one_hot_obs = torch.nn.functional.one_hot(observations[i].to(torch.int64), obs_dim)
                    
                one_hot_obs = one_hot_obs.to(torch.float32)
                torch.cat((cell_input_expand, torch.unsqueeze(torch.matmul(one_hot_obs,
                                               self.layer_vars[self.model_name+'w_state%d' % i]), 0)))# in place 操作
            else:
                assert False

        if self.input_prev_actions:
            # if True:
            if self.env_spec.combine_actions:  
                prev_action = prev_actions[0]
                for i, action_dim in enumerate(self.env_spec.orig_act_dims):
                    act = torch.fmod(prev_action, action_dim)
                    act = act.to(torch.int64)
                    act_one_hot = torch.nn.functional.one_hot(act, action_dim).to(torch.float32)
                    torch.cat((cell_input_expand, torch.unsqueeze(torch.matmul(act_one_hot, self.layer_vars[self.model_name+'w_prev_action%d' % i]), 0)))
                    prev_action = prev_action / action_dim
            else:
                for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
                    prev_action = prev_actions[i]
                    if self.env_spec.is_discrete(act_type):
                        act = torch.fmod(prev_action, act_dim)                            
                        act = act.to(torch.int64)
                        act_one_hot = torch.nn.functional.one_hot(act, act_dim).to(torch.float32)
                        torch.cat((cell_input_expand, torch.unsqueeze(torch.matmul(act_one_hot, self.layer_vars[self.model_name+'w_prev_action%d' % i]), 0)))
                    elif self.env_spec.is_box(act_type):
                        torch.cat((cell_input_expand, torch.unsqueeze(torch.matmul(
                            prev_actions[i], self.layer_vars[self.model_name+'w_prev_action%d' % i]), 0)))
                    else:
                        assert False

        cell_input = torch.sum(cell_input_expand, dim=0)
        cell_output = cell_input + self.bias

        return cell_output
