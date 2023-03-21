import torch


def Calc_state_value(model,state_batch,action_batch):
    if( len(action_batch.shape)==1):
        state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))
    else:
        state_action_values = model(state_batch).gather(1, action_batch.to(torch.int64))
    #首先说明gather函数的含义:C=A.gather(1,B)
    # 是指生成一个矩阵C，它的尺寸和B相同，其所有元素取自A
    # 元素的选取规则是这样的：
    # B总是一个非负数整数构成的矩阵，B某个位置的数字，表示A中B所在行的对应列的位置上的元素，将这个元素
    # 填到C中响应的位置上即可。更多可以去看CSDn的例子
    # 这个函数适用于分类型 agent的价值计算，其功能是：
    #action_batch表示replay池中的动作（由于是选择型AI，只能是0,1,2。。等非负数）,state_batch是对应的状态历史
    #这个函数将会计算每个action对应的价值，并返回矩阵state_action_values，大小和action一致，其值为每个时刻的动作的长期价值
    # 比方说：[1,2;3,4;5,6]是三次行动中的每个动作的评价
    #action_batch为[0;1;0]，也就是只有01值，那么最终结果就是：[1;4;5]
    #这个函数的名称就叫做：获取对应state所采取的动作的对应value(不是最高value)
    return state_action_values