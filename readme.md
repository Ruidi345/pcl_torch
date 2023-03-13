0.9.9.4
2.6.0
./CarlaUE4.sh

python trainer.py --logtostderr --batch_size=1 --env=carla \
  --validation_frequency=200 --rollout=5 --critic_weight=1.0 --gamma=0.995 \
  --clip_norm=20 --learning_rate=0.0001 \
  --replay_buffer_alpha=0.001 --norecurrent \
  --objective=trust_pcl --max_step=50 --cutoff_agent=400 --tau=0.20 --eviction=fifo \
  --max_divergence=0.001 --replay_batch_size=64 \
  --nouse_online_batch --batch_by_steps --value_hidden_layers=2 \
  --nounify_episodes --target_network_lag=0.99 --max_steer=0.8 --min_steer=-0.8\
  --clip_adv=1 --prioritize_by=step --num_steps=40000 --steer_dim=40 --throttle_dim=5\
  --noinput_prev_actions --use_target_values --tf_seed=37 --yaw_sig=0.2 --direction=0


python trainer.py --logtostderr --batch_size=1 --env=carla \
  --validation_frequency=200 --rollout=5 --critic_weight=1.0 --gamma=0.995 \
  --clip_norm=20 --learning_rate=0.0001 \
  --replay_buffer_alpha=0.001 --norecurrent \
  --objective=trust_pcl --max_step=50 --cutoff_agent=400 --tau=0.20 --eviction=fifo \
  --max_divergence=0.001 --replay_batch_size=64 \
  --nouse_online_batch --batch_by_steps --value_hidden_layers=2 \
  --nounify_episodes --target_network_lag=0.99 --max_steer=0.8 --min_steer=-0.8\
  --clip_adv=1 --prioritize_by=step --num_steps=40000 --steer_dim=40 --throttle_dim=5\
  --noinput_prev_actions --use_target_values --tf_seed=37 --yaw_sig=0.2 --direction=0

python trainer.py --logtostderr --notrain \
--path=./{saved model} --nums_record=0