import argparse
import datetime
import os
from distutils.command.build import build

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import carla_env
import controller
import env_spec
import expert_paths
import gym_wrapper
import model
import objective
import replay_buffer
from object import EnvInfo

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# args = parser.parse_args()

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging


FLAGS = flags.FLAGS
# pcl
flags.DEFINE_string('env', 'Copy-v0', 'environment name')
# pcl
flags.DEFINE_integer('batch_size', 400, 'batch size')
flags.DEFINE_integer('replay_batch_size', -1, 'replay batch size; defaults to batch_size')
flags.DEFINE_integer('num_samples', 1,
                     'number of samples from each random seed initialization')
flags.DEFINE_integer('max_step', 200, 'max number of steps to train on')
flags.DEFINE_integer('cutoff_agent', 0,
                     'number of steps at which to cut-off agent. '
                     'Defaults to always cutoff')
# pcl
flags.DEFINE_integer('validation_frequency', 50,
                     'every so many steps, output some stats')
flags.DEFINE_integer('num_steps', 200, 'number of training steps')
# pcl

flags.DEFINE_float('target_network_lag', 0.95,
                   'This exponential decay on online network yields target '
                   'network')
# pcl
flags.DEFINE_string('train_objective', 'trust_pcl',
                    'pcl/upcl/a3c/trpo/reinforce/urex')
flags.DEFINE_bool('use_online_batch', True, 'train on batches as they are sampled')
flags.DEFINE_bool('batch_by_steps', False,
                  'ensure each training batch has batch_size * max_step steps')
flags.DEFINE_bool('unify_episodes', False,
                  'Make sure replay buffer holds entire episodes, '
                  'even across distinct sampling steps')
flags.DEFINE_float('max_divergence', 0.5,
                   'max divergence (i.e. KL) to allow during '
                   'trust region optimization')
flags.DEFINE_string('sample_policy', 'None',
                    'None/softmax/spmax')

flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('clip_norm', 40.0, 'clip norm')
flags.DEFINE_float('clip_adv', 0.0, 'Clip advantages at this value.  '
                   'Leave as 0 to not clip at all.')
# pcl
flags.DEFINE_float('critic_weight', 1.0, 'critic weight')
# pcl
flags.DEFINE_float('tau', 0.025, 'entropy regularizer.'
                   'If using decaying tau, this is the final value.')
flags.DEFINE_bool('decay_tau', False,'decay_tau')
                  
flags.DEFINE_float('eps_lambda', 0.0, 'relative entropy regularizer.')
# pcl
flags.DEFINE_float('gamma', 0.9, 'discount')
# pcl
flags.DEFINE_integer('rollout', 10, 'rollout')
flags.DEFINE_bool('use_target_values', False,
                  'use target network for value estimates')

flags.DEFINE_bool('fixed_std', False,
                  'fix the std in Gaussian distributions')
flags.DEFINE_bool('input_prev_actions', False,
                  'input previous actions to policy network')
flags.DEFINE_bool('recurrent', True,
                  'use recurrent connections')
flags.DEFINE_bool('input_policy_state', True,
                  'input_policy_state')
flags.DEFINE_bool('input_time_step', False,
                  'input time step into value calucations')
flags.DEFINE_integer('value_hidden_layers', 0,
                     'number of hidden layers in value estimate')
flags.DEFINE_integer('internal_dim', 256, ' =RNN internal dim * 2')
flags.DEFINE_bool('train', True,
                  'training or evaluating')

flags.DEFINE_float('replay_buffer_alpha', 0.2, 'replay buffer alpha param')
flags.DEFINE_integer('replay_buffer_size', 80000, 'replay_buffer_size')
flags.DEFINE_string('eviction', 'rand',
                    'how to evict from replay buffer: rand/rank/fifo')
flags.DEFINE_string('prioritize_by', 'rewards',
                    'Prioritize replay buffer by "rewards" or "step"')
flags.DEFINE_integer('num_expert_paths', 0,
                     'number of expert paths to seed replay buffer with')
flags.DEFINE_integer('base', 4,
                    'dim for gym env')
# pcl
flags.DEFINE_integer('tf_seed', 42, 'random seed for tensorflow')


flags.DEFINE_string('path', 'None', './1002-1204, evaluation')
flags.DEFINE_string('load_path', 'None', './1002-1204 continue to train')

current_time = datetime.datetime.now().strftime("%m%d-%H%M")

class Trainer():
    def __init__(self):
        self.train = FLAGS.train

        if self.train:

            appendstr = ''
            if FLAGS.sample_policy == "softmax":
                appendstr = appendstr + "_sm"
            if FLAGS.decay_tau:
                appendstr = appendstr + "_dt"
            # appendstr = appendstr + FLAGS.eviction + FLAGS.prioritize_by
            log_dir = f'logs_dup_tau3/{current_time}_{FLAGS.env[0:3]}_{FLAGS.tau}_{str(FLAGS.num_expert_paths) if FLAGS.num_expert_paths > 0 else "none"}'
            self.summary_writer = tf.summary.create_file_writer(log_dir)
        else:
            print("init eval................................................")
            self.path = FLAGS.path
            assert FLAGS.path != 'None'
            with open(f'{self.path}/config.yaml') as f:
                ycfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg = FLAGS if self.train else CFG(ycfg)

        self.load_path = FLAGS.load_path
        self.num_steps = cfg.num_steps
        self.validation_frequency = cfg.validation_frequency

        # args for Controller
        self.batch_size = cfg.batch_size    
        self.env_name = cfg.env 
        self.train_objective = cfg.train_objective if self.train else "pcl"     
        self.unify_episodes = cfg.unify_episodes
        if self.unify_episodes:
            assert self.batch_size == 1
        self.use_online_batch = cfg.use_online_batch
        self.max_step = cfg.max_step        
        self.cutoff_agent = cfg.cutoff_agent        
        self.batch_by_steps = cfg.batch_by_steps      
        self.input_time_step = cfg.input_time_step # also in model
        assert not self.input_time_step or (self.cutoff_agent <= self.max_step)                 
        self.replay_batch_size = cfg.batch_size if cfg.replay_batch_size < 0 \
            else cfg.replay_batch_size
        self.target_network_lag = cfg.target_network_lag
        self.max_divergence = cfg.max_divergence
        self.tf_seed = cfg.tf_seed
        self.use_trust_region = False

        self.input_policy_state = cfg.input_policy_state
        self.input_prev_actions = cfg.input_prev_actions
        self.fixed_std = cfg.fixed_std
        self.value_hidden_layers = cfg.value_hidden_layers   
        self.internal_dim = cfg.internal_dim
        self.learning_rate = cfg.learning_rate
        self.critic_weight = cfg.critic_weight
        self.tau = cfg.tau
        self.decay_tau = cfg.decay_tau

        self.sample_policy = cfg.sample_policy
        assert self.sample_policy in ["None", "softmax", "spmax"]
        if self.sample_policy == "None":
            self.sample_policy = "spmax" if self.train_objective == 'spcl' else "softmax"            

        self.gamma = cfg.gamma
        self.rollout = cfg.rollout      
        self.eps_lambda = cfg.eps_lambda          
        self.clip_adv = cfg.clip_adv
        self.use_target_values = cfg.use_target_values
        # self.clip_norm = FLAGS.clip_norm

        self.recurrent = cfg.recurrent

        self.replay_buffer_alpha = cfg.replay_buffer_alpha
        self.replay_buffer_size = cfg.replay_buffer_size
        self.eviction = cfg.eviction 
        self.prioritize_by = cfg.prioritize_by         
        assert self.prioritize_by in ['rewards', 'step']  
        self.num_expert_paths = cfg.num_expert_paths

        self.tsaills = False  
        self.value_opt = False
        if self.train_objective == 'spcl' or self.train_objective =='trust_spcl':
            self.tsaills = True
        elif self.train_objective == 'trpo':
            self.value_opt = True
        

        self.base = cfg.base

        # self.saved_path = f'./{current_time}'

        self.hparams = dict((attr, getattr(self, attr))
                            for attr in dir(self)
                            if not attr.startswith('__') and
                            not callable(getattr(self, attr)))
        if 'summary_writer' in dir(self):
            self.hparams.pop('summary_writer')


        self.env = self.get_env()

        self.env_spec = env_spec.EnvSpec(self.env.get_one())

        self.online_model = self.get_model(target=False)
        if self.load_path != 'None':
            self.online_model.load_weights(f'{self.load_path}/model_weight')
            print("weight loaded...................")
        if not self.train:
            self.online_model.load_weights(f'{self.path}/final_model_weight')
        self.target_model = self.get_model(target=True) if 'trust' in self.train_objective \
            or self.train_objective == 'trpo' else None

        self.objective = self.get_objective()
        
        self.replay_buffer = replay_buffer.PrioritizedReplayBuffer(max_size = self.replay_buffer_size, alpha = self.replay_buffer_alpha,
                                eviction_strategy = self.eviction) if self.train_objective != 'ac' else None


        self.crtl = self.get_controller()

    def get_model(self, target):  # sourcery skip: assign-if-exp
        if self.recurrent:
            cls = model.RNNPCLModel
        else:
            cls = model.CNNPCLModel
        name = 'target' if target else 'online'

        return cls(self.env_spec, 
                    input_time_step=self.input_time_step,
                    input_policy_state=self.input_policy_state,
                    input_prev_actions=self.input_prev_actions,
                    fixed_std=self.fixed_std,
                    value_hidden_layers=self.value_hidden_layers,
                    name=name, 
                    internal_dim = self.internal_dim,
                    tsaills=self.tsaills,
                    # k=0.5, 
                    # q=2.0,
                    tau=self.tau, 
                    sample_policy = self.sample_policy)
                    # use_trust_region = self.use_trust_region

    def get_env(self):
        return gym_wrapper.Environment(self.env_name, self.base, self.batch_size)

    def get_controller(self):
        cls = controller.Controller
        return cls(batch_size=self.batch_size,
                    env_name=self.env_name,
                    env = self.env,
                    train_objective = self.train_objective,  
                    unify_episodes = self.unify_episodes,     
                    use_online_batch = self.use_online_batch,    
                    max_step = self.max_step,            
                    cutoff_agent = self.cutoff_agent, 
                    batch_by_steps = self.batch_by_steps, 
                    input_time_step = self.input_time_step,    
                    replay_batch_size = self.replay_batch_size,
                    target_network_lag = self.target_network_lag,
                    prioritize_by = self.prioritize_by,
                    # max_divergence = self.max_divergence,
                    # tf_seed = self.tf_seed,
                    online_model = self.online_model,
                    target_model = self.target_model,
                    objective = self.objective,
                    the_replay_buffer = self.replay_buffer,
                    validation_frequency = self.validation_frequency,
                    tau = self.tau,
                    total_steps =self.num_steps,
                    decay_tau = self.decay_tau,
                    value_opt = self.value_opt,
                    get_buffer_seeds = self.get_buffer_seeds,
                    train = self.train)
                    # use_trust_region=self.use_trust_region

    def get_objective(self):
        if self.train_objective == 'trust_pcl':
            cls = objective.TrustPCL
        elif self.train_objective == 'pcl':
            cls = objective.PCL
        elif self.train_objective == 'spcl':
            cls = objective.SparsePCL
        elif self.train_objective == 'ac':
            cls = objective.ActorCritic
        else:
            assert False

        return cls(learning_rate=self.learning_rate,
                    critic_weight=self.critic_weight,
                    tau=self.tau,
                    gamma=self.gamma,
                    rollout=self.rollout,
                    eps_lambda=self.eps_lambda,
                    clip_adv=self.clip_adv,
                    use_target_values=self.use_target_values,
                    max_divergence = self.max_divergence)

    def get_buffer_seeds(self):
        return expert_paths.sample_expert_paths(
            self.num_expert_paths, self.env_name, self.env_spec)

    def run(self):
        tf.compat.v1.set_random_seed(self.tf_seed)        
        losses = []
        rewards = []
        output_reward = []
        all_ep_rewards = []
        index = 1
        if self.train:
            self.crtl.set_writer(self.summary_writer)
        info_template = 'step {:d}, loss:{:.4f}, rewards:{:.4f}, ep rewards:{:.4f}, last ep reward: {:.4f}'
        print('hparams:\n%s', self.hparams_string())

        # if self.train:
        #     os.mkdir(self.saved_path)
        #     with open(self.saved_path + '/config.yaml', 'w') as f:
        #         dump_data = self.hparams
        #         yaml.dump(dump_data, f)

        for cur_step in range(self.num_steps):
            loss ,total_rewards, episode_rewards= self.crtl.train(cur_step+1)

            losses.append(loss)
            rewards.append(total_rewards)

            if cur_step % self.validation_frequency == 0 and self.train:
                all_ep_rewards.extend(episode_rewards)
                print(info_template.format\
                            (cur_step, 
                            float(np.mean(losses)), 
                            float(np.mean(rewards)),
                            float(np.mean(all_ep_rewards)) if len(all_ep_rewards)>0 else 0,
                            all_ep_rewards[-1] if len(all_ep_rewards) > 0 else 0
                            ))
                with self.summary_writer.as_default():
                    tf.summary.scalar('average episode reward', tf.reduce_mean(all_ep_rewards) if len(all_ep_rewards)>0 else 0, cur_step+1)
                    tf.summary.scalar('average episode reward(nums of epi)', tf.reduce_mean(all_ep_rewards) if len(all_ep_rewards)>0 else 0, self.env.num_episodes_played)
                    tf.summary.scalar('last episode reward', all_ep_rewards[-1] if len(all_ep_rewards) > 0 else 0, self.env.num_episodes_played)

                output_reward.append(np.mean(rewards))                
                losses = []
                rewards = []
                all_ep_rewards = []

                # if cur_step % 1000 == 0 and cur_step != 0:
                #     self.online_model.save_weights(self.saved_path + f'/model_weight{index}')
                #     index += 1



    def hparams_string(self):
        print(self.hparams)
        return '\n'.join('%s: %s' % item for item in sorted(self.hparams.items()))


class Evaluater():
    def __init__(self):
        print('building the evaluater...')
        with open('./0928-1708/config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.num_steps = cfg['num_steps']

        self.recurrent = cfg['recurrent']
        self.env_name = "carla"
        self.input_time_step = cfg['input_time_step']
        self.input_policy_state = cfg['input_policy_state']
        self.input_prev_actions = cfg['input_prev_actions']
        self.fixed_std = cfg['fixed_std']

        self.batch_size = cfg['batch_size']

        self.unify_episodes = cfg['unify_episodes']
        self.use_online_batch = cfg['use_online_batch']
        self.max_step = cfg['max_step']
        self.cutoff_agent = cfg['cutoff_agent']
        self.batch_by_steps = cfg['batch_by_steps']
        self.replay_batch_size = cfg['replay_batch_size']
        self.target_network_lag = cfg['target_network_lag']
        self.validation_frequency = cfg['validation_frequency']
        self.direction = cfg['direction']
        self.tau = cfg['tau']
        self.tsaills = cfg['tsaills']

        env_info = EnvInfo()
        if self.env_name == "carla":
            self.env = carla_env.CarlaEnv(mode=3, env_info=env_info, direction=self.direction)
        else:
            self.env = gym_wrapper.Environment(self.env_name, self.batch_size)

        self.env_spec = env_spec.EnvSpec(self.env.get_one())

        if self.recurrent:
            self.model = model.RNNPCLModel(env_spec = self.env_spec,
                                                input_time_step = self.input_time_step,
                                                input_policy_state = self.input_policy_state,
                                                input_prev_actions = self.input_prev_actions,
                                                fixed_std = self.fixed_std)
        else:
            self.model = model.CNNPCLModel(env_spec = self.env_spec,
                                                input_time_step = self.input_time_step,
                                                input_policy_state = self.input_policy_state,
                                                input_prev_actions = self.input_prev_actions,
                                                fixed_std = self.fixed_std,
                                                sample_policy = "softmax",
                                                tau=0.0,)
                                                # tsaills=self.tsaills)

        self.crtl = controller.Controller(
                                    batch_size=self.batch_size,
                                    env_name=self.env_name,
                                    env = self.env,
                                    train_objective = None,  
                                    unify_episodes = self.unify_episodes,     
                                    use_online_batch = self.use_online_batch,    
                                    max_step = self.max_step,            
                                    cutoff_agent = self.cutoff_agent, 
                                    batch_by_steps = self.batch_by_steps, 
                                    input_time_step = self.input_time_step,    
                                    replay_batch_size = self.replay_batch_size,
                                    target_network_lag = self.target_network_lag,
                                    # max_divergence = self.max_divergence,
                                    # tf_seed = self.tf_seed,
                                    online_model = self.model,
                                    target_model = None,
                                    objective = None,
                                    the_replay_buffer = None,
                                    validation_frequency = self.validation_frequency)
        print('evaluater built')

    def run(self):
        self.model.load_weights('./0928-1708/model_weight')
        print("model loaded")
        for cur_step in range(self.num_steps):
            self.crtl.evaluate(cur_step)

class CFG(object):
    def __init__(self, cfgdict):
        self.__dict__.update(cfgdict)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
