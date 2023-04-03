from PTorchEnv.StateNormalize import WelFord_Normalizer
import torch
from torch.distributions import MultivariateNormal

Normolizer=WelFord_Normalizer()
mean=torch.tensor([1.0,2.0])
cov_mat=torch.tensor([[0.5,0.0],[0.0,1.0]])
dist = MultivariateNormal(mean, cov_mat)

# Sample an action from the distribution and get its log prob
state = dist.sample()

while True:
    Normalized_state=Normolizer.normalize_state(state)
    state = dist.sample()
    print(Normalized_state)
