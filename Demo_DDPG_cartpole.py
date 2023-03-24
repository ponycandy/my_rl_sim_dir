from PTorchEnv.CartpoleTCP import CartpoleTCP
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.ContinueOpt import ContinueOpt
import random
from PyTorchTool.RLDebugger import RLDebugger
import gymnasium as gym
import numpy
from PTorchEnv.matrix_copt_tool import deepcopyMat
from tensorboardX import SummaryWriter
from PTorchEnv.RL_parameter_calc import RL_Calculator
from datetime import datetime
from NNFactory import NNFactory
import torch
from model import Critic
from model import Actor
from DDPG_Actor_Proxy import DDPG_Actor_Proxy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=ContinueOpt()
BATCHSIZE=10000
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,128,0.99,learning_rate_a=1e-4,learning_rate_c=1e-3)
envnow=CartpoleTCP(8001,"127.0.0.1")

actor_proxy=DDPG_Actor_Proxy()
actorNet=Actor(4,24,1).to(device)
actorTarget=Actor(4,24,1).to(device)
actor_proxy.setNet(actorNet)
actor_proxy.set_range([100],[0])
actor_proxy.use_eps_flag=1

actorTarget.load_state_dict(actorNet.state_dict())
Criticm=Critic(5,24,1).to(device)
CriticmTarget=Critic(5,24,1).to(device)
CriticmTarget.load_state_dict(Criticm.state_dict())

optimizer.set_NET(actorNet,actorTarget,Criticm,CriticmTarget)
initstate=[0,0,0.1,0]
lastobs=envnow.setstate(initstate)
step_done=0
total_reward=0
epoch=1
while True:
    action=actor_proxy.response(lastobs)

    obs,reward,done,info= envnow.step(action)
    total_reward+=reward

    if done:
        obs=None

    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
    lastobs=deepcopyMat(obs)
    if done or info=="truncated":
        lastobs=envnow.setstate(initstate)
        writer.add_scalar("reward",total_reward,epoch)
        epoch+=1
        total_reward=0

    actor_loss,critic_td_error=optimizer.loss_calc()

    # optimizer.updateActor(0)  这两步必须在计算完成loss之后立刻进行
    # optimizer.updateCritic(0) 否则会造成梯度的中断，故在loss_calc外面没必要再执行一次
    optimizer.SoftTargetActor(0)
    optimizer.SoftTargetCritic(0)
    step_done+=1


