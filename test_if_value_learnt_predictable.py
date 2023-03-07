from actor_proxy import actor_proxy
from PTorchEnv.Test_env_Type1 import Test_env_Type1
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.DiscreteOpt import DiscreteOpt
from tensorboardX import SummaryWriter

writer =SummaryWriter("my_log_dir")
optimizer=DiscreteOpt()
BATCHSIZE=5
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,3,0.98,1e-3)
#神经网络为D_in=1且D_out=1的网络
#神经网络输出应当拟合以下的Q value值：
#当obs=1时，输出的value等于1
#当obs=-1时，输出-1
#环境为一步式的环境，一步结束后直接进入done状态，所以，action 1 的bellman值始终是 1
envnow=Test_env_Type1()
actor=actor_proxy()
actor_target=actor_proxy()
optimizer.set_NET(actor.actor,actor_target.actor)
lastobs,reward,done,info=envnow.step([0])
step_done=0
epoch=0
total_reward=0
while True:
    action=actor.response(lastobs)
    writer.add_histogram("finaloutput",actor.actor_.save_output_layer6,step_done)
    obs,reward,done,info=envnow.step(action)
    total_reward+=reward
    if done:
        obs=None
        envnow.setstate([-1])
        total_reward=0
    print(actor.action)
    if actor.action==1:
        break
    replaybuff.appendnew(lastobs,actor.action,obs,reward) #last obs和obs打架了！！！必须detach或者深度拷贝！！
    if step_done>3:#训练过程
        loss=optimizer.loss_calc()
        loss.backward()
        optimizer.updateNetwork(0)
        optimizer.TargetNetsoftupdate(0)
        writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, step_done)
        writer.add_histogram("next state q",optimizer.record_next_state_values_musked, step_done)

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