from PTorchEnv.massel import massel

from PTorchEnv.ContinueOpt import ContinueOpt

from PTorchEnv.matrix_copt_tool import deepcopyMat

from PTorchEnv.RL_parameter_calc import RL_Calculator

from NNFactory import NNFactory

from PTorchEnv.ReplayMemory import ReplayMemory

from tensorboardX import SummaryWriter
from datetime import datetime

RL_logger=RL_Calculator()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=ContinueOpt()
BATCHSIZE=5
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,5,0.98,learning_rate_a=1e-3,learning_rate_c=1e-3)
envnow=massel()
lastobs=envnow.setstate([0])

step_done=0
epoch=0
total_reward=0

myfactory=NNFactory()
args={}
args["Net_option"]="DDPG_Net"
args["inputnum_int"]=1
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
# doesnt matter, we will not use actor_proxy
# 我们的实际bellman值：
#4.80  3.88  2.94  1.98   1   0
#神经网络必须在0 ~5上收敛至以上值
critic_proxy=proxylist["critic"]
actor_target=actor_proxy.deepCopy()
critic_target=critic_proxy.deepCopy()
optimizer.set_NET(actor_proxy.actor,actor_target.actor,critic_proxy.actor,critic_target.actor)
initstate=[0]


while True:
    # action=actor_proxy.response(lastobs)
    action=1
    obs,reward,done,info=envnow.step(action)
    total_reward+=reward

    if done or info=="speed_out":
        obs=None


    replaybuff.appendnew(lastobs,action,obs,reward)
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


