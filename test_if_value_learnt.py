from actor_proxy import actor_proxy
from PTorchEnv.massel import massel
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.DiscreteOpt import DiscreteOpt
from tensorboardX import SummaryWriter
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
optimizer=DiscreteOpt()
BATCHSIZE=5
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,5,0.98,1e-3)
# 我们的实际bellman值：
#4.80  3.88  2.94  1.98   1   0
#神经网络必须在0 ~5上收敛至以上值
envnow=massel()
actor=actor_proxy()
actor_target=actor_proxy()
optimizer.set_NET(actor.actor,actor_target.actor)
envnow.setstate([0])
lastobs=[0]
step_done=0
epoch=0
total_reward=0
while True:
    action=actor.response(lastobs)
    obs,reward,done,info=envnow.step(action)
    total_reward+=reward
    if done:
        obs=None
        envnow.setstate([0])
        total_reward=0
    replaybuff.appendnew(lastobs,actor.action,obs,reward) #last obs和obs打架了！！！必须detach或者深度拷贝！！
    if step_done>3:#训练过程
        loss=optimizer.loss_calc()
        loss.backward()
        optimizer.updateNetwork(0)
        optimizer.TargetNetsoftupdate(0)
        writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, step_done)
        writer.add_histogram("next state q",optimizer.record_next_state_values_musked, step_done)
        # writer.add_histogram("next state q",optimizer.record_next_state_values_musked, step_done)
        # writer.add_histogram("weight",actor.actor_.layerx.weight,step_done)
        # writer.add_histogram("bias",actor.actor_.layerx.bias,step_done)
        writer.add_scalar("reward",total_reward,step_done)
        writer.add_scalar("loss/train",loss,step_done)
    if done or epoch>10:
        lastobs=[0]
        envnow.setstate([0])
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