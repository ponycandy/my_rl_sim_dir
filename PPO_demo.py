import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
import torch
import numpy as np

import gym
# import roboschool
from tensorboardX import SummaryWriter

from PPO import PPO

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "CartPole-v1"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################


    env = gym.make(env_name)






    replaybuff=PPO_Buffer()
    optimizer = PPO()
    optimizer.set_Replaybuff(replaybuff,gamma,lr_actor,lr_critic)

    actor_proxy=PPO_Actor_Proxy()
    actorNet=Actor_Softmax(4,24,2)
    criticM=Critic_PPO(4,24,1)
    actor_proxy.setNet(actorNet,criticM)
    actor_proxy.set_action_dim(2)
    actor_proxy.setActFlag("Discrete")
    actor_proxy.set_range([1,1],[0,0])
    optimizer.setNet(actorNet,criticM,actor_proxy)



    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    state = env.reset()
    lastobs=TensorTypecheck(state[0])
    current_ep_reward = 0
    epoch=1
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
    # training loop
    while True:
            # select action with policy
        action=actor_proxy.response(lastobs)
        obs, reward, done, truncated ,_= env.step(action)
        replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
        lastobs=TensorTypecheck(obs)
        if done or truncated:
            replaybuff.ResetNotify()#only notify once!!
            state = env.reset()
            lastobs=TensorTypecheck(state[0])
            writer.add_scalar("reward",current_ep_reward,epoch)
            epoch+=1
            current_ep_reward = 0

#这里使用logger记录会比较好
            #因为提升是迅速的
        time_step +=1
        current_ep_reward += reward
        optimizer.update()







if __name__ == '__main__':

    train()







