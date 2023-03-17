from PTorchEnv.PushingBoxTCP import PushingBoxTCP
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.ContinueOpt import ContinueOpt
import random
from PyTorchTool.RLDebugger import RLDebugger
from PTorchEnv.matrix_copt_tool import deepcopyMat
from tensorboardX import SummaryWriter
from PTorchEnv.RL_parameter_calc import RL_Calculator
from datetime import datetime
from NNFactory import NNFactory
import torch
RL_logger=RL_Calculator()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=ContinueOpt()
BATCHSIZE=100
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,100,0.9,learning_rate_a=1e-3,learning_rate_c=1e-3)
envnow=PushingBoxTCP(8001,"127.0.0.1")
myfactory=NNFactory()
args={}
args["Net_option"]="DDPG_Net"
args["inputnum_int"]=2
args["outputnum_int"]=1
args["num_layers"]=2
args["n_units_l0"]=8
args["n_units_l1"]=10
args["len_act"]=1
args["len_state"]=1
args["len_output"]=1
args["action_n_units_l0"]=10
args["state_n_units_l0"]=10
args["output_n_units_l0"]=10
args["Merge_results"]=10


proxylist=myfactory.create_agent(args)
actor_proxy=proxylist["actor"]
actor_proxy.set_range([10],[0])
actor_proxy.use_eps_flag=1
critic_proxy=proxylist["critic"]
actor_target=actor_proxy.deepCopy()
critic_target=critic_proxy.deepCopy()
optimizer.set_NET(actor_proxy.actor,actor_target.actor,critic_proxy.actor,critic_target.actor)
initstate=[1,1]
lastobs=envnow.setstate(initstate)
step_done=0
total_reward=0
epoch=1
while True:
    action=actor_proxy.response(lastobs)

    obs,reward,done,info=envnow.step(action)
    total_reward+=reward

    if done or info=="speed_out":
        obs=None


    replaybuff.appendnew(lastobs,actor_proxy.action,obs,reward)
    if done or info=="speed_out":
        lastobs=envnow.setstate(initstate)
        if step_done>BATCHSIZE+10 :#训练过程,只在每次完成一个epoch之后进行，并不是每一步都执行
            actor_loss,critic_td_error=optimizer.loss_calc()
            epoch+=1

            # optimizer.updateActor(0)  这两步必须在计算完成loss之后立刻进行
            # optimizer.updateCritic(0) 否则会造成梯度的中断，故在loss_calc外面没必要再执行一次
            optimizer.SoftTargetActor(0)
            optimizer.SoftTargetCritic(0)

            writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, epoch)
            writer.add_scalar("reward",total_reward,epoch)
            writer.add_scalar("critic_td_error/train",critic_td_error,epoch)
            writer.add_scalar("actor_loss/train",critic_td_error,epoch)
    else:
        lastobs=deepcopyMat(obs)
    step_done+=1


