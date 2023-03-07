#尝试替换现有的Replaybuffer(把数据都加载在内存中，对于存储空间来说无法接受)
#10000条数据，就是80000个Byte，就是800KB(呃，那好像还可以哈)

import time
import random
import torch
from PTorchEnv.ReplayMemory import ReplayMemory
m_replay=ReplayMemory(10000)
listmy=["VARCHAR(255)" for _ in range(3)]
time_start = time.time()  # 记录开始时间
for i in range(0,10000):
    print(i)
    m_replay.appendnew(torch.tensor([[1,2,3],[4,5,6]]),torch.tensor([[1,2,3],[4,5,6]]),torch.tensor([[1,2,3],[4,5,6]]),0)
# function()   执行的程序
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

time_start = time.time()  # 记录开始时间
m_replay.sample(1000)
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
# 随机读取一条数据，在完成排序后的效率很高：

# SELECT * FROM  testtable WHERE COMMON_ID >= ((SELECT MAX(COMMON_ID) FROM testtable)-(SELECT  MIN(COMMON_ID) FROM testtable)) * RAND() + (SELECT MIN(COMMON_ID) FROM testtable)       LIMIT 1;
