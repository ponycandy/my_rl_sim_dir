class PlayBuffer():
    def __init__(self):
        pass
    def pushback(self,observation,reward,action):
        tup = (observation,action, reward)
        self.batch_state_action_reward_tuples.append(tup)
