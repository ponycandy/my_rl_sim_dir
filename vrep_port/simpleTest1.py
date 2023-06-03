import os
import glob
import time
from Boardlogger import Boardlogger
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
import PTorchEnv.model
from PTorchEnv.Typechecker import TensorTypecheck
from DroneEnv import DroneEnv


from PPO import PPO


####### initialize environment hyperparameters ######
env=DroneEnv(23005)

#vrep的接口和TCP的接口还不一样
#TCP的接口只要考虑两端就行了，
#但是vrep的接口用的是zmq，所以需要考虑到其它的东西
#但是我们还是以这个例子开始训练比较好




replaybuff=PPO_Buffer()
optimizer = PPO()
optimizer.set_Replaybuff(replaybuff,0.99, 0.0003, 0.001)

actor_proxy=PPO_Actor_Proxy()
actorNet=PTorchEnv.model.Actor_Drone()
criticM=PTorchEnv.model.Critic_PPO(12,128,1)
actor_proxy.setNet(actorNet,criticM)

actor_proxy.set_action_dim(4)
actor_proxy.setActFlag("Continuous")
actor_proxy.set_range([5,5,5,5],[0,0,0,0])
actor_proxy.decayInterval=200
optimizer.setNet(actorNet,criticM,actor_proxy)




time_step = 0

lastobs = env.randominit()
current_ep_reward = 0
epoch=1
rllogger=Boardlogger()
# training loop
while True:
    # select action with policy
    action=actor_proxy.response(lastobs)
    obs,reward,done,info= env.step(action)
    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
    lastobs=TensorTypecheck(obs)
    if done or info== "truncated": #drone任务中没有truncated的情况,只有一个终点
        replaybuff.ResetNotify()#only notify once!!
        lastobs = env.randominit()
        rllogger.log_per_step_scalr(current_ep_reward,"epoc_reward")
        epoch+=1
        current_ep_reward = 0
    time_step +=1
    current_ep_reward += reward
    optimizer.update()









