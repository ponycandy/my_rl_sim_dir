#Policy gradient算法，参照DDPG实现：
#https://github.com/sherlockHSY/Reinforcement_learning_with_pytorch/blob/90c4f302b588bbf8be7962aaaa7f61c0234fb8d9/model_free/DDPG/DDPG.py#L100
# 已通过的测试:value_learnt,value_learnt_predictable,discount_correct,
# log_prob记录错误可能性：
# 已经排除
#
# 也就是考虑各个数据的尺寸不对的问题：
# 已排除
#
# 怀疑:actorloss的计算图不正确：
# 未排除
from PTorchEnv.Category_Func import Calc_state_value
import torch
import torch.optim as optim
from torchviz import make_dot


class ProbOpt:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def set_Replaybuff(self,buf,batchsize,reward_decay_rate,learning_rate_a,learning_rate_c,updatesteps):
        self.replaybuff=buf
        self.n_updates_per_iteration=80 #超参，表示一次升级循环计算的次数
        # 按照经验选取为80
        self.BATCHSIZE=batchsize
        self.epoch=0
        self.eps_clip=0.2  # As recommended by the paperd q
        self.GAMA=reward_decay_rate
        self.LR_a=learning_rate_a
        self.LR_c=learning_rate_c
        self.mse_loss = torch.nn.MSELoss()
        self.Advantage_Calc_method=0
        self.timesteps_per_batch = updatesteps            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode 实际上只要保证truncated极限小于这个值就行，大多数实验都满足这个条件
    def set_NET(self,actorNet,criticNet,actor_proxy):
        self.actorNet=actorNet
        self.criticNet=criticNet
        self.actor_proxy=actor_proxy
        self.optimizer_a = optim.AdamW(self.actorNet.parameters(), lr=self.LR_a, amsgrad=True)
        self.optimizer_c = optim.AdamW(self.criticNet.parameters(), lr=self.LR_c, amsgrad=True)

        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()
    def evaluate(self, batch_obs,batch_acts):
    # 计算所有状态对应的V值预测值，动作概率值，以及顺便的计算分布的熵值
    # 由于每次更新后都需要重算，这一部分无法使用buff中存储的值，必须在线计算
        if self.actor_proxy.act_flag==1:#连续
            self.actor_proxy.continuous_response(batch_obs)
            dist=self.actor_proxy.action[2]
            action_logprobs = dist.log_prob(batch_acts)
            dist_entropy = dist.entropy()
            state_values = self.criticNet(batch_obs)

        else:
            self.actor_proxy.Discrete_response(batch_obs)
            dist=self.actor_proxy.action[2]
            action_logprobs = dist.log_prob(batch_acts.squeeze(1))
            dist_entropy = dist.entropy()
            state_values = self.criticNet(batch_obs)
        # if self.has_continuous_action_space:
        #     action_mean = self.actor(state)
        #
        #     action_var = self.action_var.expand_as(action_mean)
        #     cov_mat = torch.diag_embed(action_var).to(device)
        #     dist = MultivariateNormal(action_mean, cov_mat)
        #
        #     # For Single Action Environments.
        #     if self.action_dim == 1:
        #         action = action.reshape(-1, self.action_dim)
        # else:
        #     action_probs = self.actor(state)
        #     dist = Categorical(action_probs)


        return action_logprobs, state_values, dist_entropy
    def compute_rtgs(self, batch_rews):
        # 此时，距离losscalc前的最后一个epoch有可能既没有truncated，也没有terminated，所以需要确认一下当前的
        # ep_rews是不是为空（为空则已经被resetNotify过了，否则就是处在运行状态中被强制打断了）
        if len(self.replaybuff.ep_rews)==0:
            pass
        else:
            self.replaybuff.ResetNotify()
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.GAMA
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs
    def CalcAdvantage(self,method):
        if method==0:#按照TD误差计算优势函数
            #计算按照序列排列的value值,这个值是真实value值
            batch_rtgs=self.compute_rtgs(self.replaybuff.batch_rews)
            # 计算预测的V值
            self.replaybuff.batch_lastobs=torch.stack(self.replaybuff.batch_lastobs,1).squeeze(0)
            value = self.criticNet(self.replaybuff.batch_lastobs) #已经验证，确实是使用上一状态量

            # 这里和DQN类似，终点是不存在avantage的，所以恒等于0，不需要用终点量进行判定
            # 有一个问题，batch_rtgs是包含终点量的
            # 这里需要debug一下了，确保上面两个值对应顺序正确
    #         例子：
    # obs
    # [1 1 1 0] [1 1 0]
    # last obs
    # [X 1 1 1] [X 1 1]
    # ep_rews:
    # [1,1,1,0]  [1,1,0]
    # batch_rews:
    # [[1,1,1,0],[1,1,0]]
    # Reset两次
    #
    # [1,1,0]   [1,1,1,0]
    #
    # batch_rtgs:
    # [ 2.71  1.9 1 0 1.9 1 0]
    #
    # 这个是从0时刻开始，每一步的真实value
    # episode按照正向顺序填充（）
    # 也就是episode 2 的所有正向序列，接在episode 1的正向序列后面
    #
    # 注意到一点,初始的lastobs!!，对应上面的第一个价值
    # 后面的价值，依次对应下一个lastobs
    # 所以，我们是用lastobs去计算的，但是，对于终点量的value值，我们应该用obs而不是last_obs去算
            A_k = batch_rtgs.unsqueeze(1).to(self.device) - value.detach()
            return A_k,batch_rtgs
        if method==1:#按照GAE方法计算优势函数
            A=0
            return A,A

    def loss_calc(self):
        if  self.replaybuff.total_timestep<self.timesteps_per_batch:  #这是执行优化的条件，batch被填满了
            loss1 = torch.zeros(1, requires_grad=True)
            return loss1
        # 计算优势函数A,真实V值系列
        self.epoch+=1
        loss = torch.zeros(1, requires_grad=True)
        advantage,batch_rtgs=self.CalcAdvantage(self.Advantage_Calc_method)
        # Normalize advantages另一种方法是直接normalize rewards都可以稳定训练过程
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        self.replaybuff.batch_acts=torch.cat(self.replaybuff.batch_acts,0)
        self.replaybuff.batch_log_probs=torch.stack(self.replaybuff.batch_log_probs)
        # 以下做简单的说明：1.每一轮更新都需要用更新后的网络对各个动作的概率，各个状态的价值进行重新计算
        # 所以需要evaluate包含一些其他的代码
        for _ in range(self.n_updates_per_iteration):
            logprobs, state_values, dist_entropy = self.evaluate(self.replaybuff.batch_lastobs, self.replaybuff.batch_acts)
            ratios = torch.exp(logprobs.unsqueeze(1) - self.replaybuff.batch_log_probs.detach())


            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage

            # final loss of clipped objective PPO,参照强化学习精要，添加了熵约束的优化
            # 此为actor的loss
            # loss = (-torch.min(surr1, surr2) - 0.01 * dist_entropy.unsqueeze(1)).mean()
            loss = (-torch.min(surr1, surr2)).mean()
            # take gradient step
            self.optimizer_a.zero_grad()
            loss.backward()
            self.optimizer_a.step()

            critic_loss = torch.nn.MSELoss()(state_values, batch_rtgs.unsqueeze(1))
            #这里critic追踪的是真实的V值，而不是TDerror的V值，所以是无偏的，不需要targetNet用来纠偏
            self.optimizer_c.zero_grad()
            critic_loss.backward()
            self.optimizer_c.step()
        self.replaybuff.clearbuffer()
        return loss