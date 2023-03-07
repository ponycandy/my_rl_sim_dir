from criticNet import criticNet
class critic_proxy():
    def __init__(self):
        self.critic=criticNet(1,1,1)
    def set_member(self):
        pass
    def predict(self,vector):
        output=self.critic.forward(vector)
        self.value=output
        return  self.value