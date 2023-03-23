#Policy gradient算法，参照DDPG实现：
#https://github.com/sherlockHSY/Reinforcement_learning_with_pytorch/blob/90c4f302b588bbf8be7962aaaa7f61c0234fb8d9/model_free/DDPG/DDPG.py#L100
# 已通过的测试:value_learnt,value_learnt_predictable,discount_correct,actor_trained
from PTorchEnv.Category_Func import Calc_state_value
import torch
import torch.optim as optim
class ContinueOpt:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def set_Replaybuff(self,buf,batchsize,reward_decay_rate,learning_rate_a,learning_rate_c):
        self.replaybuff=buf
        self.BATCHSIZE=batchsize
        self.GAMA=reward_decay_rate
        self.LR_a=learning_rate_a
        self.LR_c=learning_rate_c
        self.mse_loss = torch.nn.MSELoss()
    def set_NET(self,actorNet,actor_targetNet,criticNet,critic_targetNet):
        self.actorNet=actorNet
        self.actor_targetNet=actor_targetNet
        self.criticNet=criticNet
        self.critic_targetNet=critic_targetNet
        self.optimizer_a = optim.AdamW(self.actorNet.parameters(), lr=self.LR_a, amsgrad=True)
        self.optimizer_c = optim.AdamW(self.criticNet.parameters(), lr=self.LR_c, amsgrad=True)

        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()

    def get_sample(self):
        state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=self.replaybuff.get_Batch_data(self.BATCHSIZE)
        return state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states
    def loss_calc(self):
        if  self.replaybuff.recordinglength< self.BATCHSIZE:
            self.optimizer_c.zero_grad()
            self.optimizer_a.zero_grad()

            loss1 = torch.zeros(1, requires_grad=True)
            loss2=loss1
            return loss1,loss2
        if(hasattr(self, 'replaybuff')):
            state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=self.get_sample()
        else:
            print("optimizer dont have a buffer yet!")
            return
        # actor的损失函数
        all_action = self.actorNet(state_batch)
        all_Q_value = self.criticNet(state_batch, all_action)
        actor_loss = -torch.mean(all_Q_value) #optimizer总是按照最小化loss的方向前进，在这里就是最大化reward
        self.optimizer_a.zero_grad()
        actor_loss.backward(retain_graph=True)

        self.updateActor()
        #注意，这里不能够在外面backward，否则会和下面的backward发生计算图串联！！
        #源代码也是
        #critic的TD error
        #需要考虑非终点量的方法
        Q_value_next_state = torch.zeros(self.BATCHSIZE,device=self.device).unsqueeze(1)
        if self.replaybuff.empty_nextstate_flag==0:
            with torch.no_grad():
                # 需要考虑到action是多维度的
                next_state_action = self.actor_targetNet(non_final_next_states)
            #next_state_action与non_final_next_states同维度，下面的criticNet只输出非终点量的价值预测，所以不需要non_final_mask
        # 下面的式子需要补足终点量的bellman值
                Q_value_next_state[non_final_mask] = self.critic_targetNet(non_final_next_states, next_state_action)
        else:
            pass
        q_bellman_target = reward_batch.to(torch.float32) + self.GAMA * Q_value_next_state  #对于terminal state需要补全为0
        self.record_expected_state_action_values=q_bellman_target
        q_eval = self.criticNet(state_batch.to(torch.float32), action_batch.to(self.device).to(torch.float32))
        critic_td_error = self.mse_loss(q_bellman_target,q_eval)
        # critic_td_error=torch.nn.SmoothL1Loss(q_bellman_target,q_eval)
        self.optimizer_c.zero_grad()
        critic_td_error.backward()

        self.updateCritic()

        return actor_loss,critic_td_error

    def updateActor(self,update_method=0):
            #总是使用优化器
        torch.nn.utils.clip_grad_value_(self.actorNet.parameters(), 100) #梯度裁剪，一种防止梯度爆炸的优化策略，非必要
        self.optimizer_a.step()
        self.optimizer_a.zero_grad()
        return
    def updateCritic(self,update_method=0):
        #总是使用优化器
        torch.nn.utils.clip_grad_value_(self.criticNet.parameters(), 100) #梯度裁剪，一种防止梯度爆炸的优化策略，非必要
        self.optimizer_c.step()
        self.optimizer_c.zero_grad()
        return

    def SoftTargetActor(self,update_method=0):
        if(update_method==0):#此方法将actor网络的权重乘以一个系数后加和到目标网络上
            TAU=0.1
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
    def SoftTargetCritic(self,update_method=0):
        if(update_method==0):#此方法将actor网络的权重乘以一个系数后加和到目标网络上
            TAU=0.1
            target_net_state_dict = self.critic_targetNet.state_dict()
            policy_net_state_dict = self.criticNet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.critic_targetNet.load_state_dict(target_net_state_dict)
            return
        if(update_method==1):#每N步执行一次完全替换，中间的判断不变，替换计时由外部提供
            target_net_state_dict = self.critic_targetNet.state_dict()
            policy_net_state_dict = self.criticNet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            self.critic_targetNet.load_state_dict(target_net_state_dict)
            return