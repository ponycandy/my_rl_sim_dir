import copy
import torch
class DDPG_critic_Proxy():
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def setNet(self,Net):
        self.actor=Net
    def deepCopy(self):
        return copy.deepcopy(self)
    def response(self,vector):
        output=self.predict(vector)
        self.action=output
        return output
    def randresponse(self):
        output=self.random_action()
        self.action=output
        return self.chooseaction(output)