from actor_proxy import actor_proxy
from PTorchEnv.CartpoleTCP import CartpoleTCP
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.DiscreteOpt import DiscreteOpt
import random
from PyTorchTool.RLDebugger import RLDebugger
from PTorchEnv.matrix_copt_tool import deepcopyMat
from tensorboardX import SummaryWriter
from PTorchEnv.RL_parameter_calc import RL_Calculator
from datetime import datetime
import torch
RL_logger=RL_Calculator()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=DiscreteOpt()
BATCHSIZE=10000
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,128,0.9,1e-4)
envnow=CartpoleTCP(8001,"127.0.0.1")
actor=actor_proxy()
actor.actor_.writer=writer
actor.use_eps_flag=0
actor.EPS_DECAY=1000
actor_target=actor_proxy()
actor_target.actor_.load_state_dict(actor.actor_.state_dict())
optimizer.set_NET(actor.actor,actor_target.actor)
initstate=[0,0,0.5*(random.random()-0.5),0]
# initstate=[0,0,0.5*(0.1-0.5),0]
lastobs=envnow.randominit()

step_done=0
total_reward=0
epoch=1
while True:
    action=actor.response(lastobs)

    obs,reward,done,info=envnow.step(action)
    total_reward+=reward

    if done or info=="speed_out":
        # print("the action is:",action,"angle now:",obs[1,0])

        obs=None


    replaybuff.appendnew(lastobs,actor.action,obs,reward)
    if done or info=="speed_out":
        lastobs=envnow.randominit()
        if step_done>250 :#训练过程,只在每次完成一个epoch之后进行，并不是每一步都执行
            loss=optimizer.loss_calc()
            loss.backward()
            optimizer.updateNetwork(0)
            epoch+=1
            # ploterr.add_a_point(epoch,total_reward)

            optimizer.TargetNetsoftupdate(0)
            actor.use_eps_flag=0
            with torch.no_grad():
                variance=RL_logger.calc_Residual_Varriance_Iterative(optimizer.record_expected_state_action_values[0,0],
                                                                     optimizer.record_next_state_values_musked.unsqueeze(1)[0,0])
                writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, epoch)
                writer.add_histogram("next state q",optimizer.record_next_state_values_musked, epoch)
                if epoch <100:
                    pass
                else:
                    writer.add_histogram("Residual_Varriance",variance, epoch)
                writer.add_scalar("reward",total_reward,epoch)
                writer.add_scalar("loss/train",loss,epoch)
                actor.use_eps_flag=1
                total_reward=0
    else:
        lastobs=deepcopyMat(obs)
    step_done+=1


#这样一来，总的代码量就大量减小了
#同时，需要重复的低级细节：如何解包压缩经验池，如何计算目标函数，如何升级权值，就可以被隐藏了
#目前暴露的细节就是：1.我们可以选择合适选取经验池进行训练 2.选择网络升级的方法
#封包的细节：1.off-policy的实施 2.targetNet的实施
#要考虑的是通用性，现有的方法应该能够通用于一切value based agent

#当前模型已经分析有效
#试一下pendulum经典环境，学习非常之快，几乎一个迭代就能够获得目标行为



