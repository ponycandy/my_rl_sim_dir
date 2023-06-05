# import time
#
# from ray.rllib.algorithms.ppo import PPOConfig
# import gym
# from datetime import datetime
# from tensorboardX import SummaryWriter
# from ray.tune.logger import pretty_print
# config = PPOConfig()
# config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
# config = config.resources(num_gpus=1)
# config = config.framework("torch")
# config = config.rollouts(num_rollout_workers=4)
# print(config.to_dict())
# # Build a Algorithm object from the config and run 1 training iteration.
# #要想训练多次，需要设置多个循环...
# algo = config.build(env="CartPole-v1")
# epoch=1
# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# start=time.time()
# algo.train()
# end=time.time()
# print(end-start,'s')
#
# rd=0
# while rd<480:
#     start = time.time()
#     result=algo.train()
#     end = time.time()
#     print(end - start, 's')
#     rd+=1
#     writer.add_scalar("reward", result['episode_reward_mean'], epoch)
#     epoch+=1
#
# envnow = gym.make("CartPole-v1", render_mode="human")
# obs=envnow.reset()
# obs=obs[0]
# steps=0
# while True:
#     action=algo.compute_single_action(obs)
#     obs, reward, done, truncated, _ = envnow.step(action)
#     steps+=1
#     if done or truncated:
#         state = envnow.reset()
from PTorchEnv.HumanoidGym import HumanoidGym
import gym
envnow = gym.make('HumanoidBulletEnv-v0')