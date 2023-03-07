class Game_Player():
    def __init__(self):
        self.step_done=0
        self.maxstep=0
        pass
    def play_game(self):
        action=actorNow.act(last_observation)
    # step the environment and get new measurements
        observation, reward, done, info = envnow.step(action)

        total_reward=total_reward+reward
        if done:
            observation=None
        self.step_done+=1
        buff.push(last_observation, action, observation, reward)
        last_observation=observation
    def judge_time(self):
        if len(memory) < BATCH_SIZE:
            return 0#not right time to train
        if self.step_done>self.maxstep:
            self.step_done=0
            return 1
    def train(self):
        pass