import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG import DDPGagent
from utils import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env = NormalizedEnv(gym.make("Pendulum-v1"))
env2=gym.make("Pendulum-v1")
agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(50):
    state ,infos= env2.reset()
    noise.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, terminated, truncated ,_= env2.step(action)
        done=terminated or truncated
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                         np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()