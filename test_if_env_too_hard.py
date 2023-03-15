

from actor_proxy import actor_proxy
from PTorchEnv.PushingBoxTCP import PushingBoxTCP
# from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.DiscreteOpt import DiscreteOpt
from PTorchEnv.matrix_copt_tool import deepcopyMat
import random
from datetime import datetime
from tensorboardX import SummaryWriter
from PyTorchTool.RLDebugger import RLDebugger
#如果这个环境测试可以通过，说明模型和算法已经能够通过最简单的复杂环境进行学习了
#如果到目前为止的所有测试通过而目标环境不工作（这个RL和目标RL的唯一区别就是所用的env）
#说明目标环境的reward设计不够合理，或者采样不合理，或者环境不可学习
#下一步测试建议手写算法，检验环境是否MDP，算法是否可学
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=DiscreteOpt()
BATCHSIZE=1000
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,256,0.9,1e-3)
envnow=PushingBoxTCP(8001,"127.0.0.1")
actor=actor_proxy()
actor.actor_.writer=writer
actor.use_eps_flag=1
actor_target=actor_proxy()
optimizer.set_NET(actor.actor,actor_target.actor)
initstate=[1,1]
# initstate=[0,0,0.5*(0.1-0.5),0]
lastobs=envnow.setstate(initstate)

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
        initstate=[1,1]
        lastobs=envnow.setstate(initstate)

    replaybuff.appendnew(lastobs,actor.action,obs,reward)
    if done or info=="speed_out":
        if step_done>BATCHSIZE+10 :#训练过程
            loss=optimizer.loss_calc()
            loss.backward()
            writer.add_histogram("gradient check",actor.actor_.layer1.weight.grad, epoch)
            optimizer.updateNetwork(0)
            epoch+=1
            # ploterr.add_a_point(epoch,total_reward)

            optimizer.TargetNetsoftupdate(0)
            actor.use_eps_flag=0
            # actor.response(lastobs)
            writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, epoch)
            writer.add_histogram("next state q",optimizer.record_next_state_values_musked, epoch)
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



