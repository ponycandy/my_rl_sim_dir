import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
from PTorchEnv.pushingbox2D_tcp import PushingBox2DTCP,PushingBox2DTCP_exact
from PyTorchTool.FileManager import FileManager
from PyTorchTool.Boardlogger import Boardlogger
from PyTorchTool.Terminator_class import Terminator_class
import torch
import numpy as np

import gym
# import roboschool
from tensorboardX import SummaryWriter

from PPO import PPO
preatrained=False
use_pretrainmodel=True
using_critic=False
if preatrained==True:
    ####### initialize environment hyperparameters ######
    envnow=PushingBox2DTCP(8001,"127.0.0.1")







    replaybuff=PPO_Buffer()
    optimizer = PPO()
    optimizer.set_Replaybuff(replaybuff,0.99, 0.0003, 0.001)


    actor_proxy=PPO_Actor_Proxy()
    actorNet=Actor(4,24,2)
    criticM=Critic_PPO(4,24,1)
    actor_proxy.setNet(actorNet,criticM)
    actor_proxy.set_action_dim(2)
    actor_proxy.setActFlag("Continuous")
    actor_proxy.set_range([2,2],[0,0])
    actor_proxy.decayInterval=200
    optimizer.setNet(actorNet,criticM,actor_proxy)
    optimizer.updateinterval=300



    time_step = 0

    initstate=[1,0,1,0]
    lastobs=envnow.setstate(initstate)

    current_ep_reward = 0
    epoch=1
    rllogger=Boardlogger()
    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
    # training loop
    terminate=Terminator_class(156,10)
    while True:
        # select action with policy
        action=actor_proxy.response(lastobs)
        obs,reward,done,info= envnow.step(action)
        replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
        lastobs=TensorTypecheck(obs)
        if done or info=="truncated":
            replaybuff.ResetNotify()#only notify once!!
            lastobs=envnow.setstate(initstate)
            rllogger.log_per_step_scalr(current_ep_reward,"ep_reward")
            if(terminate.judge(current_ep_reward)):
                break
            epoch+=1
            current_ep_reward = 0

        #这里使用logger记录会比较好
        #因为提升是迅速的
        time_step +=1
        current_ep_reward += reward
        optimizer.update()


    manager=FileManager()
    manager.save_model_out(actor_proxy.actor,"actor_net_pretrained")
    manager.save_model_out(actor_proxy.critic,"critic_net_pretrained")
    print("done")

else:
    envnow=PushingBox2DTCP_exact(8001,"127.0.0.1")







    replaybuff=PPO_Buffer()
    optimizer = PPO()
    optimizer.set_Replaybuff(replaybuff,0.99, 0.0003, 0.001)


    actor_proxy=PPO_Actor_Proxy()
    actorNet=Actor(4,24,2)
    criticM=Critic_PPO(4,24,1)
    if(use_pretrainmodel):
        state_dict = torch.load('../SuperVislearning/actor_net_pretrained.pt')
        actorNet.load_state_dict(state_dict)
        if using_critic:
            state_dict = torch.load('critic_net_pretrained.pt')
            criticM.load_state_dict(state_dict)
    actor_proxy.setNet(actorNet,criticM)
    actor_proxy.set_action_dim(2)
    actor_proxy.setActFlag("Continuous")
    actor_proxy.set_range([10,10],[0,0])
    actor_proxy.decayInterval=200
    optimizer.setNet(actorNet,criticM,actor_proxy)
    optimizer.updateinterval=300



    time_step = 0

    initstate=[1,0,1,0]
    lastobs=envnow.setstate(initstate)

    current_ep_reward = 0
    epoch=1
    rllogger=Boardlogger()
    while True:
        # select action with policy
        action=actor_proxy.response(lastobs)
        # action=10*actorNet(lastobs.to(torch.float32))
        obs,reward,done,info= envnow.step(action.cpu())
        replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
        lastobs=TensorTypecheck(obs)
        if done or info=="truncated":
            replaybuff.ResetNotify()#only notify once!!
            lastobs=envnow.setstate(initstate)
            rllogger.log_per_step_scalr(current_ep_reward,"ep_reward")
            epoch+=1
            current_ep_reward = 0

        #这里使用logger记录会比较好
        #因为提升是迅速的
        time_step +=1
        current_ep_reward += reward
        optimizer.update()





