from ReplayMemory import ReplayMemory

buff=ReplayMemory(7)
buff.appendnew(0,0,-1,-1)
buff.appendnew(-1,0,-2,-1)
buff.appendnew(-2,0,-3,-1)
buff.appendnew(-3,0,-4,-1)
buff.appendnew(-4,0,-5,-1)

state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=buff.get_Batch_data(5)
print(1)