#可靠性原理，尽量使用他人制造的无错误的轮子，这是真正工程师的思维（站在巨人的肩膀上，而不是手搓一个，后者没有意义）
#算法原网页：https://yulizi123.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/

from SumTree import SumTree
from collections import namedtuple, deque
import torch
import random
import numpy as np
from PTorchEnv.Typechecker import TensorTypecheck
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class Prioritized_Replaybuffer():
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.99  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    def __init__(self,capacity):
        self.tree = SumTree(capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.empty_nextstate_flag=0
        pass
    def sample(self,batchsize):
        pass
    def appendnew(self,lastobs,act,state,reward):
        transition=Transition(TensorTypecheck(lastobs),TensorTypecheck(act),TensorTypecheck(state),TensorTypecheck(reward))
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p
        #报错，注意Prioritized_Replaybuffer要求必须填满整个capacity后才能开始训练
    def get_Batch_data(self,BATCH_SIZE):
        n=BATCH_SIZE
        b_memory=[]
        b_idx,  ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
#确保训练前，本buffer已经填满，否则会出现不可预料的错误
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i]= idx
            b_memory.append(data)
        # returnlist_1=[ b_idx, b_memory, ISWeights]
        self.weight_record=torch.from_numpy(ISWeights).to(self.device)
        self.index_recorded=b_idx
        #上面是prioritized部分，下面是取样部分，可以仿照replaybuffer

        batch = Transition(*zip(*b_memory)) #b_memory should be a list of Transisions
        #这一步将上面transions里面的state,action,nextstate，rewards取出来，各构成一个向量
        #顺序保持压入的时候的一致性
        #式子从上面的变成下面的,例子如下：
        #[Transition(state=13, action=14, next_state=None, reward=16), Transition(state=17, action=18, next_state=19, reward=20), Transition(state=9, action=10, next_state=None, reward=12)...
        # Transition(state=(13, 17, 9, 5, 1), action=(14, 18, 10, 6, 2), next_state=(None, 19, None, 7, 3), reward=(16, 20, 12, 8, 4))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        #如果所有的状态都是非终止态，则non_final_mask=[true,true,true.....]
        #如果某一个状态为终止状态，则返回该状态为flase non_final_mask=[true,False,true.....]
        if all(i is None for i in batch.next_state):
            self.empty_nextstate_flag=1
            non_final_next_states=[]
        else:
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
        #将所有的非终止状态缝起来维度可能要小于下面三个
        #上面两个东西的作用是，实现非终点状态的补位：
        # next_state_values = torch.zeros(BATCH_SIZE)
        # next_state_values[non_final_mask] = targetNet(non_final_next_states).max(1)[0]
        # 以上面的指令为例，next_state_values的尺寸大于non_final_next_states，但是等于non_final_mask
        #non_final_mask中的False项使得上述赋值过程，直接将这个值给0，从而使得两边相等
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(self.device)
        #这三个都是全系列的，不考虑下一步是否终结，尺寸始终为Batchsize
        return state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states
        #返回的三个值含义是：bacth在树中的index，bacth数据本身，batch的权重
        #问题是，bacth的权重如何对网络的优化产生影响
    #     self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
    # 此处是计算损失函数的过程，可见是要将整个loss都乘以权重，换言之，操作只在loss计算中进行
    def batch_update(self, tree_idx, abs_errors):#abs_errors为n*1矩阵，二维tensor
        #这里看来abserror应该是各个差值的abs值
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.cpu(), self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        count=0
        for ti in tree_idx:
            self.tree.update(ti, ps[count,0].item())
            count+=1
    #保证对外接口不变！！