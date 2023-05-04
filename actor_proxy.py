from actorNet import actorNet
from PyTorchTool.Category_proxy import Category_proxy
import torch
import copy

class DQN(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

class actor_proxy(Category_proxy):
    def __init__(self):
        super(actor_proxy, self).__init__(2) #首先调用父类的初始化函数进行初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_=DQN(4,2).to(self.device)
        # self.actor_=torch.load('pretrain_cartpole.pt').to(self.device)
        self.setNet(self.actor_)

        self.writer=0
        pass
    def set_member(self):
        pass
    def response(self,vector):
        output=self.predict(vector)
        self.action=output
        return self.chooseaction(output)
    def randresponse(self):
        output=self.random_action()
        self.action=output
        return self.chooseaction(output)
    def chooseaction(self,num):
        if num==0:
        #左移
            force=-30
            return force
        if num==1:
            force=30
            return force
    def deepCopy(self):
        target_net_proxy= copy.deepcopy(self)
        target_net_proxy.actor_.load_state_dict(self.actor_.state_dict())
        return target_net_proxy
