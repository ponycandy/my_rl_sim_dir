from actor_proxy import actor_proxy
import gymnasium as gym
from PTorchEnv.CartpoleTCP import CartpoleTCP
from PTorchEnv.massel import massel
from PTorchEnv.PPO_ReplayMemory import PPO_ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.ProbOpt import ProbOpt
import random
from PyTorchTool.RLDebugger import RLDebugger
from PTorchEnv.matrix_copt_tool import deepcopyMat
from tensorboardX import SummaryWriter
from PTorchEnv.RL_parameter_calc import RL_Calculator
from datetime import datetime
import torch
from model import Critic
from model import Actor
from model import Actor_Softmax
from model import Critic_PPO
from PPO_Actor_Proxy import PPO_Actor_Proxy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RL_logger=RL_Calculator()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=ProbOpt()
BATCHSIZE=10000
replaybuff=PPO_ReplayMemory(BATCHSIZE)

optimizer.set_Replaybuff(replaybuff,128,0.99,1e-4,1e-3,5)
# envnow=CartpoleTCP(8001,"127.0.0.1")
envnow = gym.make("CartPole-v1")
envnow=massel()

actor_proxy=PPO_Actor_Proxy()
actorNet=Actor_Softmax(1,24,2).to(device)


actor_proxy.setNet(actorNet)
actor_proxy.set_range([2],[0])
actor_proxy.setActFlag("Discrete")
actor_proxy.use_eps_flag=1
actor_proxy.set_action_dim(1)

Criticm=Critic_PPO(1,24,1).to(device)

# critic只用来判断value值，不用来判断Q值，所以无需动作的输入
# 这就是Q值和V值区别的最直接例子
optimizer.set_NET(actorNet,Criticm,actor_proxy)

initstate=[0]
envnow.setstate([0])
lastobs = torch.tensor(initstate, dtype=torch.float32, device=device).unsqueeze(0)
step_done=0
total_reward=0
epoch=1
while True:
    action=actor_proxy.response(lastobs)

    observation,reward,done,info=envnow.step(action)
    total_reward+=reward
    reward = torch.tensor([reward], device=device)
    # done = terminated or truncated
    if done:
        obs=None
    else:
        obs = torch.tensor(observation, dtype=torch.float32, device=device)


    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)

    if done:
        # PPO不可避免的需要进行terminated检测，所以需要补上下面这个非常规行:总是在进行重置的时候呼出下面这个repllay的函数
        replaybuff.ResetNotify()
        envnow.setstate([0])
        lastobs = torch.tensor(initstate, dtype=torch.float32, device=device).unsqueeze(0)
        print(total_reward)
        epoch+=1
        total_reward=0
    else:
        lastobs=deepcopyMat(obs)

    loss=optimizer.loss_calc()  #将critic和actor的update分开，这里应该就没有什么soft update了吧




    step_done+=1


# 各个优化器和模块的更改不导致上面主程序的更改



