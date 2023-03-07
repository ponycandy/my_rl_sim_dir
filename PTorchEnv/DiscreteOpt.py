#类描述：要求对离散动作输出的agent进行优化,
#即agent输出为各个动作的value值,agent是一个value based的离散动作智能体
#该算法默认使用的优化准则：1.targetNet准则 2.off-policy准则
#要求以下API:loss=Optimizer.losscalc();Optimizer.updateNetwork(int update_method)
from PTorchEnv.Category_Func import Calc_state_value
import torch
import torch.optim as optim
class DiscreteOpt:
    def __init__(self):
        pass
    def set_Replaybuff(self,buf,batchsize,reward_decay_rate,learning_rate):
        self.replaybuff=buf
        self.BATCHSIZE=batchsize
        self.GAMA=reward_decay_rate
        self.LR=learning_rate
        self.criterion = torch.nn.SmoothL1Loss()
        # 确定是否使用优先级replaybuffer，确保切换buffer做对比验证的时候不需要更改此处代码
        if hasattr(self.replaybuff,"batch_update"):
            self.use_prioritized_buffer=1
        else:
            self.use_prioritized_buffer=0
    def set_NET(self,actorNet,actor_targetNet):
        self.actorNet=actorNet
        self.actor_targetNet=actor_targetNet
        self.optimizer = optim.AdamW(self.actorNet.parameters(), lr=self.LR, amsgrad=True)
        self.optimizer.zero_grad()
    def get_sample(self):
        state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=self.replaybuff.get_Batch_data(self.BATCHSIZE)
        return state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states
    def loss_calc(self):
        if(hasattr(self, 'replaybuff')):
            state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=self.get_sample()
        else:
            print("optimizer dont have a buffer yet!")
            return
        if(hasattr(self, 'actorNet') and hasattr(self, 'actor_targetNet')):
            state_action_values=Calc_state_value(self.actorNet,state_batch,action_batch)
        else:
            print("optimizer net is not set yet!")
            return
        next_state_values = torch.zeros(self.BATCHSIZE)
        with torch.no_grad():
            if self.replaybuff.empty_nextstate_flag==0:  #进行test_if测试算法或者其它变量的影响时，防止报错
                next_state_values[non_final_mask] = self.actor_targetNet(non_final_next_states).max(1)[0]
                #这种赋值方法就是，使用一个tensor来赋值，这个tensor是bool的，在对应true的地方，将值拷贝过来，False地方，拷贝0值
                #例如：non_final_mask=[ True,  True, False,  True,  True]
                #上式右侧为[0.5126, 0.5458, 0.6562, 0.7063]
                #则左侧为[0.5126, 0.5458, 0.0000, 0.6562, 0.7063]
                #刚好为DQN定义的终点奖励为0的状态
            else:
                pass


        expected_state_action_values = (next_state_values.unsqueeze(1) * self.GAMA) + reward_batch
        self.record_expected_state_action_values=expected_state_action_values
        self.record_next_state_values_musked=next_state_values
        if self.use_prioritized_buffer==1:

            expected_state_action_values_weighted=torch.multiply(self.replaybuff.weight_record,expected_state_action_values)
            state_action_values_weighted=torch.multiply(self.replaybuff.weight_record,state_action_values)
        #此时是prioritized replay buff，需要更新经验池中所有transison的权值，此部分无法与DiscreteOPT解耦
        #原本计算的是0.5*(state_action_values-expected_state_action_values)**2，现在，我们给每一个
        #样本乘以其对应的权值，变成：0.5*(weight*(state_action_values-expected_state_action_values))**2
        #然后求和，这就是prioritized方法的原理
            loss = self.criterion(state_action_values_weighted, expected_state_action_values_weighted)
            with torch.no_grad():
                abs_error=abs(expected_state_action_values-state_action_values)
                self.replaybuff.batch_update(self.replaybuff.index_recorded,abs_error)
                #这一步根据loss更新各个样本的权值
            return loss
        if self.use_prioritized_buffer==0:
            # loss=0.5*(state_action_values-expected_state_action_values)**2
            #求出每一个TD误差后，要对总误差求和,也就是上面的向量各项求和，可以使用torch的内建函数解决：

            loss = self.criterion(state_action_values, expected_state_action_values)
            return loss
            #此时按照一般的计算方法计算loss




    def updateNetwork(self,update_method):
        if(update_method==0):#此方法激活使用优化器
            torch.nn.utils.clip_grad_value_(self.actorNet.parameters(), 100) #梯度裁剪，一种防止梯度爆炸的优化策略，非必要
            self.optimizer.step()
            self.optimizer.zero_grad()
            return
        if(update_method==1):#此方法激活手动优化，极不推荐
            for param in self.actorNet.parameters():
                param.data = param.data-param.grad*self.LR
            self.actorNet.zero_grad()
    def TargetNetsoftupdate(self,update_method):
        if(update_method==0):#此方法将actor网络的权重乘以一个系数后加和到目标网络上
            TAU=0.005
            target_net_state_dict = self.actor_targetNet.state_dict()
            policy_net_state_dict = self.actorNet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.actor_targetNet.load_state_dict(target_net_state_dict)
            return
        if(update_method==1):#每N步执行一次完全替换，中间的判断不变，替换计时由外部提供
            target_net_state_dict = self.actor_targetNet.state_dict()
            policy_net_state_dict = self.actorNet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            self.actor_targetNet.load_state_dict(target_net_state_dict)
            return