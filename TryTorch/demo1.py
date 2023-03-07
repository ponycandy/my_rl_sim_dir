from M_actpx import M_actpx
from M_cripx import critic_proxy
from PTorchEnv.PushingBoxTCP import PushingBoxTCP
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.ContinueOpt import ContinueOpt
optimizer=ContinueOpt()
BATCHSIZE=100
replaybuff=ReplayMemory(BATCHSIZE)
optimizer.set_Replaybuff(replaybuff,25,0.98,1e-4,1e-4)
actor=M_actpx()
actor.set_epsilon(1)
actor_target=M_actpx()
critic=critic_proxy()
critic_target=critic_proxy()
envnow=PushingBoxTCP(8001,"127.0.0.1")
optimizer.set_NET(actor.actor,actor_target.actor,critic.critic,critic_target.critic)
initstate=[1 ,1]
envnow.setstate(initstate)
lastobs=initstate
step_done=0
epoch_steps=0
while True:
    action=actor.response(lastobs)
    print(action)
    obs,reward,done,info=envnow.step(action)
    if done:
        obs=None
        envnow.setstate(initstate)
    replaybuff.appendnew(lastobs,actor.action,obs,reward)
    if step_done>30:#训练过程
        actor_loss,critic_td_error=optimizer.loss_calc()
        #这里面已经包含了backward和update两部，这里无法解耦，因为计算图无法分离
        #简单来说就是框架问题
        optimizer.SoftTargetActor()
        optimizer.SoftTargetCritic()

    if done:
        lastobs=initstate
        envnow.setstate(initstate)
        epoch_steps=0
    else:
        lastobs=obs
    step_done+=1
    epoch_steps+=1


