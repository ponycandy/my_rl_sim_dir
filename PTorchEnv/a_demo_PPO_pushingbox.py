import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
from PTorchEnv.PushingBoxTCP import PushingBoxTCP
import torch
import numpy as np

import gym
# import roboschool
from tensorboardX import SummaryWriter

from PPO import PPO


####### initialize environment hyperparameters ######
envnow=PushingBoxTCP(8001,"127.0.0.1")







replaybuff=PPO_Buffer()
optimizer = PPO()
optimizer.set_Replaybuff(replaybuff,0.99, 0.0003, 0.001)


actor_proxy=PPO_Actor_Proxy()
actorNet=Actor(2,24,1)
criticM=Critic_PPO(2,24,1)
actor_proxy.setNet(actorNet,criticM)
actor_proxy.set_action_dim(1)
actor_proxy.setActFlag("Continuous")
actor_proxy.set_range([10],[0])
actor_proxy.decayInterval=200
optimizer.setNet(actorNet,criticM,actor_proxy)
optimizer.updateinterval=300



time_step = 0

initstate=[1,0]
lastobs=envnow.setstate(initstate)

current_ep_reward = 0
epoch=1
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# training loop
while True:
    # select action with policy
    action=actor_proxy.response(lastobs)
    obs,reward,done,info= envnow.step(action)
    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
    lastobs=TensorTypecheck(obs)
    if done or info=="truncated":
        replaybuff.ResetNotify()#only notify once!!
        lastobs=envnow.setstate(initstate)
        writer.add_scalar("reward",current_ep_reward,epoch)
        print(current_ep_reward)
        epoch+=1
        current_ep_reward = 0

    #这里使用logger记录会比较好
    #因为提升是迅速的
    time_step +=1
    current_ep_reward += reward
    optimizer.update()









