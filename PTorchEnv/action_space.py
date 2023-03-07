# import torch
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# for i in range(100):
#     loss = i
#     writer.add_scalar("loss",loss,i)
# writer.close()


from tensorboardX import SummaryWriter
import numpy as np
writer = SummaryWriter()
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)
writer.close()
