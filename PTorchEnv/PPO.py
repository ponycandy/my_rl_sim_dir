import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


################################## PPO Policy ##################################

class PPO:
    def __init__(self):



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 80
        self.stepdone=1
        self.updateinterval=4000


        self.MseLoss = nn.MSELoss()
    def set_Replaybuff(self,replaybuff,GAMA,LR_Actor,LR_Critic):
        self.buffer = replaybuff
        self.gamma = GAMA
        self.LR_Actor=LR_Actor
        self.LR_Critic=LR_Critic
    def SetUpdateInterval(self,Interval):
        self.updateinterval=Interval
    def setNet(self,actor,critic,actorproxy):
        self.actorNet=actor
        self.criticNet=critic
        self.actor_proxy=actorproxy
        self.optimizer = torch.optim.Adam([
            {'params': self.actorNet.parameters(), 'lr': self.LR_Actor},
            {'params': self.criticNet.parameters(), 'lr': self.LR_Critic}
        ])



    def CalcAdvatange(self,method):
        if method==0:#按照TD误差计算优势函数
            _,_,_,old_state_values=self.buffer.get_Batch_data()
            rewards=self.Compute_RTGS()
            advantages = rewards.detach() - old_state_values.detach()
            return advantages,rewards
    def Compute_RTGS(self):#计算真实价值
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        return rewards
    def update(self):
        if self.stepdone % self.updateinterval != 0:
            self.stepdone+=1
            return 0

        self.stepdone=1
        old_states,old_actions,old_logprobs,old_state_values=self.buffer.get_Batch_data()
        #获取所有buffer数据
        advantages,rewards=self.CalcAdvatange(0)
        #计算优势函数
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy=self.evaluate(old_states,old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2)+ 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            #这个损失函数将critic和actor的损失函数放一块了
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        # self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def evaluate(self,old_states, old_actions):
        if self.actor_proxy.act_flag==1:#连续
            mean = self.actorNet(old_states)
            # Create our Multivariate Normal Distribution
            dist = MultivariateNormal(mean, self.actor_proxy.cov_mat)
            # Sample an action from the distribution and get its log prob
            logprobs = dist.log_prob(old_actions)
            state_values=self.criticNet(old_states)
            dist_entropy=dist.entropy()
        #
        else:
            action_probs = self.actorNet(old_states)
            dist = Categorical(action_probs)

            logprobs = dist.log_prob(old_actions)
            state_values=self.criticNet(old_states)
            dist_entropy=dist.entropy()

        # logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        return logprobs, state_values, dist_entropy





