import numpy as np
import torch
from  Prioritized_Replaybuffer import Prioritized_Replaybuffer


replaybuff=Prioritized_Replaybuffer(5)
replaybuff.appendnew(1,1,None,1)
replaybuff.appendnew(1,1,None,1)
replaybuff.appendnew(-1,1,None,-1)
replaybuff.appendnew(-1,-1,None,1)
replaybuff.appendnew(1,-1,None,-1)

state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=replaybuff.get_Batch_data(5)

print("getall!")
abs_error=torch.tensor([[1.0,2.0,3.0,4.0,5.0]]).t()
replaybuff.batch_update(replaybuff.index_recorded,abs_error)

#验证高loss项在下次更容易选中