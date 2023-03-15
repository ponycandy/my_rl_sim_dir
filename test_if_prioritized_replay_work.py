# import numpy as np
# import torch
# from  Prioritized_Replaybuffer import Prioritized_Replaybuffer
#
#
# replaybuff=Prioritized_Replaybuffer(4)
# # replaybuff.appendnew(1,1,None,1)  #1
# replaybuff.appendnew(1,1,None,1) #1
# replaybuff.appendnew(-1,1,None,-1)  #2
# replaybuff.appendnew(-1,-1,None,1)  #3
# replaybuff.appendnew(1,-1,None,-1)  #4
#
# state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=replaybuff.get_Batch_data(4)
#
# print("getall!")
# abs_error=torch.tensor([[0.1,0.2,0.3,0.4]]).t()
# #首先,abs_error只有在1以下（abs_error_upper）时才会有用
# replaybuff.batch_update(replaybuff.index_recorded,abs_error)
#
# #验证高loss项在下次更容易选中
# count1=0
# count2=0
# count3=0
# count4=0
# count5=0
# while True:
#     # abs_error=torch.tensor([[0.1,0.2,0.3,0.4]]).t()
#     state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=replaybuff.get_Batch_data(1)
#     if state_batch[0,0]== 1 and action_batch[0,0]==1:
#         abs_error=torch.tensor([[0.1]]).t()
#         count1+=1
#     if state_batch[0,0]==-1 and action_batch[0,0]==1:
#         abs_error=torch.tensor([[0.2]]).t()
#         count2+=1
#     if state_batch[0,0]==-1 and action_batch[0,0]==-1:
#         abs_error=torch.tensor([[0.3]]).t()
#         count3+=1
#     if state_batch[0,0]==1 and action_batch[0,0]==-1:
#         abs_error=torch.tensor([[0.4]]).t()
#         count4+=1
#     print(count1,count2,count3,count4)
#     replaybuff.batch_update(replaybuff.index_recorded,abs_error)
# #在搜寻树中，各个leaf的loss会接近于输入的loss，也就是abs_error
# #但是，前提是，上面的error均显著小于1，否则必须要估计error的上界，不然，会导致所有的节点loss均为1
# #理论上，没有太大问题，也可以验证count4显著大于另外两个了，的确加大了大baserror的选中概率
# #那么，问题可能出现在loss函数上面？
# #也就是isweight的计算上面？



from actor_proxy import actor_proxy
from PTorchEnv.TestType_env2 import TestType_env2
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.DiscreteOpt import DiscreteOpt
from tensorboardX import SummaryWriter

writer =SummaryWriter("my_log_dir")
optimizer=DiscreteOpt()
BATCHSIZE=20
# replaybuff=Prioritized_Replaybuffer(BATCHSIZE)
replaybuff=ReplayMemory(BATCHSIZE)
replaybuff=Prioritized_Replaybuffer(BATCHSIZE)

optimizer.set_Replaybuff(replaybuff,10,0.98,1e-3)
#神经网络为D_in=1且D_out=2的网络
#神经网络输出应当拟合以下的Q value值：
#当obs=1时，动作1输出的value等于1，动作-1输出的value值为-1
#当obs=-1时，与上面相反
#环境为一步式的环境，一步结束后直接进入done状态
#由于每次采取的都是权值比较高的运动，reward=-1的采样概率会不断减小
#所以可以看到，到了后期，-1的值无法被学习！！！但是+1的值已经被完美的拟合了

#但是，在采取prioritized replay方法时，情况就不一样了
#loss更大的样本会被优先学习，所以-1样本会被优先采取。通过对比我们应该能够发现这个规律
#尽管如此，当前的replaybuffer确实可以改进：权值较大的经验应该被更久的保留（deque中不被那么容易舍弃
# ）
envnow=TestType_env2()
actor=actor_proxy()
actor_target=actor_proxy()
optimizer.set_NET(actor.actor,actor_target.actor)
lastobs,reward,done,info=envnow.step([0])
step_done=0
epoch=0
total_reward=0
while True:
    action=actor.response(lastobs)
    obs,reward,done,info=envnow.step(action)
    print(reward)
    total_reward+=reward
    if done:
        obs=None
        envnow.setstate([-1])
        total_reward=0
    # if action==-1:
    #     break
    replaybuff.appendnew(lastobs,actor.action,obs,reward) #last obs和obs打架了！！！必须detach或者深度拷贝！！
    if step_done>20:#训练过程
        loss=optimizer.loss_calc()
        loss.backward()
        optimizer.updateNetwork(0)
        optimizer.TargetNetsoftupdate(0)
        writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, step_done)
        writer.add_histogram("next state q",optimizer.record_next_state_values_musked, step_done)
        writer.add_histogram("output",actor.actor_.save_output_layer6,step_done)
        # writer.add_histogram("weight",actor.actor_.layerx.weight,step_done)
        # writer.add_histogram("bias",actor.actor_.layerx.bias,step_done)
        writer.add_scalar("reward",total_reward,step_done)
        writer.add_scalar("loss/train",loss,step_done)
    if done or epoch>10:
        lastobs,reward,done,info=envnow.step(0)
        epoch=0

    else:
        lastobs=obs.copy() #必须深拷贝！！！
    step_done+=1
    epoch+=1

#这样一来，总的代码量就大量减小了
#同时，需要重复的低级细节：如何解包压缩经验池，如何计算目标函数，如何升级权值，就可以被隐藏了
#目前暴露的细节就是：1.我们可以选择合适选取经验池进行训练 2.选择网络升级的方法
#封包的细节：1.off-policy的实施 2.targetNet的实施
#要考虑的是通用性，现有的方法应该能够通用于一切value based agent

#当前模型已经分析有效
#试一下pendulum经典环境，学习非常之快，几乎一个迭代就能够获得目标行为





from actor_proxy import actor_proxy
from PTorchEnv.PushingBoxTCP import PushingBoxTCP
# from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.DiscreteOpt import DiscreteOpt
from PTorchEnv.matrix_copt_tool import deepcopyMat
import random
from datetime import datetime
from tensorboardX import SummaryWriter
from PyTorchTool.RLDebugger import RLDebugger
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()
optimizer=DiscreteOpt()
BATCHSIZE=1000
replaybuff=Prioritized_Replaybuffer(BATCHSIZE)
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
        envnow.setstate(initstate)


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
            initstate=[1,1]
            lastobs=envnow.setstate(initstate)
        lastobs=envnow.setstate(initstate)

    else:
        lastobs=deepcopyMat(obs)
    step_done+=1


#已经验证prioritized replaybuff有效，并且学习epochs减少了约三分之一，算是极其重大的飞跃



