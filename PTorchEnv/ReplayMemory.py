from collections import namedtuple, deque
import torch
import random
import numpy as np
from PTorchEnv.Typechecker import TensorTypecheck
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.recordinglength=0
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.empty_nextstate_flag=0
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def appendnew(self,lastobs,act,state,reward):
        #请注意，这里的act只能是动作的index，而不能够是动作本身
        self.recordinglength+=1
        self.memory.append(Transition(TensorTypecheck(lastobs),TensorTypecheck(act),TensorTypecheck(state),TensorTypecheck(reward)))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def get_Batch_data(self,BATCH_SIZE):
        transitions = self.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
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
        state_batch = torch.cat(batch.state).to(torch.float32)
        action_batch = torch.cat(batch.action).to(torch.float32)
        reward_batch = torch.cat(batch.reward).to(self.device)
        #这三个都是全系列的，不考虑下一步是否终结，尺寸始终为Batchsize
        return state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states
