from NetworkCollection.TreeCritic import criticNet
class critic_proxy():
    def __init__(self):
        self.critic=criticNet(1,1,1)
    def set_member(self):
        pass
    def predict(self,state,action):
        output=self.critic.forward(state,action)
        self.value=output
        return  self.value