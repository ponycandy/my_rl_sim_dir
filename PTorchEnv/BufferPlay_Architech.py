from ReplayMemory import ReplayMemory

buff=ReplayMemory(5)


buff.appendnew([0,0],[5,0],[0,0],1)
buff.appendnew([0,1],[0,0],[0,0],2)
buff.appendnew([0,2],[0,0],[0,0],3)
buff.appendnew([0,0],[0,3],None,4)
buff.appendnew([0,0],[4,0],[0,0],5)
buff.get_Batch_data(5)