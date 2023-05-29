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
from PPO import PPO


class PPO_Single_instance():
    def __init__(self):
        pass
    def set_env(self,env):
        self.envnow=PushingBox2DTCP(8001,"127.0.0.1")
    def train_func(self):
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
        lastobs=self.envnow.setstate(initstate)

        current_ep_reward = 0
        epoch=1
        rllogger=Boardlogger()
        terminate=Terminator_class(156,10)
        while True:
            # select action with policy
            action=actor_proxy.response(lastobs)
            obs,reward,done,info= self.envnow.step(action)
            replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
            lastobs=TensorTypecheck(obs)
            if done or info=="truncated":
                replaybuff.ResetNotify()#only notify once!!
                lastobs=self.envnow.setstate(initstate)
                rllogger.log_per_step_scalr(current_ep_reward,"ep_reward")
                epoch+=1
                current_ep_reward = 0

            #这里使用logger记录会比较好
            #因为提升是迅速的
            time_step +=1
            current_ep_reward += reward
            optimizer.update()