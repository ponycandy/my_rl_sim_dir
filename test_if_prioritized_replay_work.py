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