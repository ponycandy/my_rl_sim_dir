from massel import massel
import numpy as np
from  policyiteration_Net.actor_proxy import actor_proxy
from actor_proxy import actor_proxy
from ReplayMemory import ReplayMemory
def one_loop(actor,env):
    done = False
    total_reward = 0
    while True:
        last_observation = env.Getobservation([0])
        action=actor.act(last_observation)
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        total_reward=total_reward+reward
        if done:
            break
        return total_reward
envnow=massel()
actorNow=actor_proxy()
envnow.setstate([0])
# one_loop(actorNow,envnow)
buff=ReplayMemory()
last_observation = envnow.Getobservation([0])

done = False
total_reward = 0
while True:

    action=actorNow.act(last_observation)
    # step the environment and get new measurements
    observation, reward, done, info = envnow.step(action)
    total_reward=total_reward+reward
    buff.push(last_observation, action, observation, reward)
    last_observation=observation
    if done:
        break
