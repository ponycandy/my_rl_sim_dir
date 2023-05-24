import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
from PTorchEnv.CartpoleTCP import CartpoleTCP
import torch
import numpy as np

import gym
# import roboschool
from tensorboardX import SummaryWriter

from PPO import PPO


####### initialize environment hyperparameters ######
env=CartpoleTCP(8001,"127.0.0.1")






replaybuff=PPO_Buffer()
optimizer = PPO()
optimizer.set_Replaybuff(replaybuff,0.99, 0.0003, 0.001)

actor_proxy=PPO_Actor_Proxy()
actorNet=Actor(4,128,1)
criticM=Critic_PPO(4,128,1)
actor_proxy.setNet(actorNet,criticM)

actor_proxy.set_action_dim(1)
actor_proxy.setActFlag("Continuous")
actor_proxy.set_range([20],[0])
actor_proxy.decayInterval=200
optimizer.setNet(actorNet,criticM,actor_proxy)




time_step = 0

lastobs = env.randominit()
current_ep_reward = 0
epoch=1
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# training loop
while True:
    # select action with policy
    action=actor_proxy.response(lastobs)
    obs,reward,done,info= env.step(action)
    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
    lastobs=TensorTypecheck(obs)
    if done or info== "truncated":
        replaybuff.ResetNotify()#only notify once!!
        lastobs = env.randominit()
        writer.add_scalar("reward",current_ep_reward,epoch)
        epoch+=1
        current_ep_reward = 0
    time_step +=1
    current_ep_reward += reward
    optimizer.update()









