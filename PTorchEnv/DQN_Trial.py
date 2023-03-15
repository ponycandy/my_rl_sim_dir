from Trial_base import Trial_base
from DiscreteOpt import DiscreteOpt
import torch
from Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.matrix_copt_tool import deepcopyMat
#我们总是允许重写本类，这样sample_hyper_parameters和trainingloop都具备很强大的可变性
#可以方便的插入可视化贺debug断点工具
class DQN_Trial(Trial_base):
    def __init__(self,inputnum_int,outputnum_int,output_act_list):
        super(DQN_Trial,self).__init__()
        self.inputnum_int=inputnum_int
        self.outputnum_int=outputnum_int
        self.output_act_list=output_act_list
        self.Batchsize=1000#default ,can be changed
        self.trainingbatch=250#default,can be changed
        self.set_pointee(self)
    def set_external_command(self,command_body):
        self.external_command=command_body
    def State_Set_Command(self):
        return self.external_command.state_init()
    # 这部分总是需要改写
    def sample_hyper_parameters(self,trial):
        #目前考虑的优化参数：神经网络的结构，学习率，贪心率
        #实际上来说，我们希望的自动化学习系统应该是类似于控制系统
        #允许我们实时（在训练中）调整参数（如探索率），而不只是在每一轮
# 训练结束后才重调参数，所以，目前的技术状态只是一种中间态，我们可以将这项技术
# 与实时参数调节技术相结合，极大提高系统的产出效率
        num_layers = trial.suggest_int("num_layers", 1, 3)#神经元的总层次数目
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)#学习率
        EPS_DECAY=trial.suggest_int("EPS_DECAY", 1000, 10000, log=True)#贪心衰减率
        args={}

        args["num_layers"]=num_layers
        args["learning_rate"]=learning_rate
        args["EPS_DECAY"]=EPS_DECAY
        for i in range(num_layers):
            n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            args["n_units_l{}".format(i)]=n_units#单层神经网络的神经元个数
        #一些DQN的其它选项，这部分是解耦的关键
        args["Net_option"]="DQN_Net"
        args["inputnum_int"]=self.inputnum_int
        args["outputnum_int"]=self.outputnum_int
        args["output_act_list"]=self.output_act_list
        return args
    def TrainingLoop(self,n_episodes_to_train,agent,args):
        #DQN的TraingingLoop我们希望是可以更改的
        #因此，使用可变参数列表会极大的削弱这种可变更性
        #我的建议是，保持下面的这种繁复形式，作为系统的default设置
        #需要添加新的写法的时候，直接复制本类，写一个DQN_Trial_2的类
        #因为我们需要足够多的细节暴露出来，这样的可变性最强
        optimizer=DiscreteOpt()
        BATCHSIZE=self.Batchsize
        replaybuff=Prioritized_Replaybuffer(BATCHSIZE)
        optimizer.set_Replaybuff(replaybuff,self.trainingbatch,0.9,agent.learning_rate)
        # discount率可以改变，请自行根据需求重写本类，此处只是一个默认配置
        envnow=self.envnow
        actor=agent
        actor.use_eps_flag=1
        actor_target=agent.deepCopy()
        optimizer.set_NET(actor.actor,actor_target.actor)
        initstate=envnow.randominit()
        lastobs=initstate

        step_done=0
        total_reward=0
        epoch=1
        while epoch<n_episodes_to_train:
            action=actor.response(lastobs)
            obs,reward,done,info=envnow.step(action)
            total_reward+=reward

            if done or info=="speed_out":
                obs=None


            replaybuff.appendnew(lastobs,actor.action,obs,reward)
            if done or info=="speed_out":
                lastobs=envnow.randominit()
                if step_done>BATCHSIZE+10 :#训练过程,只在每次完成一个epoch之后进行，并不是每一步都执行
                    loss=optimizer.loss_calc()
                    loss.backward()
                    # writer.add_histogram("gradient check",actor.actor_.layer1.weight.grad, epoch)
                    optimizer.updateNetwork(0)
                    epoch+=1

                    optimizer.TargetNetsoftupdate(0)
                    # actor.use_eps_flag=0
                    # with torch.no_grad():
                    #     variance=RL_logger.calc_Residual_Varriance_Iterative(optimizer.record_expected_state_action_values[0,0],
                    #                                                          optimizer.record_next_state_values_musked.unsqueeze(1)[0,0])
                    #     writer.add_histogram("Bellman",optimizer.record_expected_state_action_values, epoch)
                    #     writer.add_histogram("next state q",optimizer.record_next_state_values_musked, epoch)
                    #     if epoch <300:
                    #         pass
                    #     else:
                    #         writer.add_histogram("Residual_Varriance",variance, epoch)
                    #     writer.add_scalar("reward",total_reward,epoch)
                    #     writer.add_scalar("loss/train",loss,epoch)
                    #     actor.use_eps_flag=1
                    #     total_reward=0
            else:
                lastobs=deepcopyMat(obs)
            step_done+=1
