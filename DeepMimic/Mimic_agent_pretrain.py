import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from swarm_proxy import SwarmProxy
from PTorchEnv.model import Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
from SwarmTCP import SwarmTCP
from PyTorchTool.Boardlogger import Boardlogger
from tensorboardX import SummaryWriter

from PPO import PPO


####### initialize environment hyperparameters ######
envnow=SwarmTCP(8001,"127.0.0.1")







replaybuff=PPO_Buffer()
optimizer = PPO()
optimizer.set_Replaybuff(replaybuff,0.99, 0.0003, 0.001)


actor_proxy=SwarmProxy()
actorNet=Actor(9,128,6)
criticM=Critic_PPO(9,128,1)
actor_proxy.setNet(actorNet,criticM)
actor_proxy.set_action_dim(6)
actor_proxy.setActFlag("Continuous")
actor_proxy.set_range([2,1,2,1,2,1],[0,0,0,0,0,0]) #速度和角速度间隔
actor_proxy.decayInterval=200
optimizer.setNet(actorNet,criticM,actor_proxy)
optimizer.updateinterval=300



time_step = 0

lastobs=envnow.reset()

current_ep_reward = 0
epoch=1
m_log=Boardlogger()
while True:
    # select action with policy
    action=actor_proxy.response(lastobs)
    obs,reward,done,info= envnow.step(action)
    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
    lastobs=TensorTypecheck(obs)
    if done or info=="truncated":
        replaybuff.ResetNotify()#only notify once!!
        lastobs=envnow.reset()
        epoch+=1
        m_log.log_average_per_step_scalr(current_ep_reward,"averge_rpisode_reward",10)
        current_ep_reward = 0

    #这里使用logger记录会比较好
    #因为提升是迅速的
    time_step +=1

    current_ep_reward += reward
    optimizer.update()









