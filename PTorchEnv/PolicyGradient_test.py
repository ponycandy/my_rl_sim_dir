from PTorchEnv.policyiteration_Net.actor_proxy import actor_proxy
from critic_proxy import critic_proxy
from Littlesruare import Littlesruare

init_state=[0,-1.0] #-1补位,重置信号
control=[0,-2.0]#-2补位，步进信号
envnow=Littlesruare(8001,"127.0.0.1")
envnow.setstate(init_state)

actornow=actor_proxy()

criticnow=critic_proxy()


while True:
    done = False
    total_reward = 0
    round_number = 1

    envnow.setstate(init_state)
    last_observation = envnow.Getobservation(init_state)
    action = environment.action_space.sample()

    # step the environment and get new measurements
    observation, reward, done, info = environment.step(action)
    observation = preprocess(observation)
    number_steps = 1

    while not done:
        observation_delta = observation - last_observation
        last_observation = observation
        update_probability = actornow.actor.forward(observation_delta)[0]
        if np.random.uniform() < update_probability:
            action = UP_ACTION
        else:
            action = DOWN_ACTION

        # step the environment and get measurements
        observation, reward, done, info = environment.step(action)
        observation = preprocess(observation)
        total_reward += reward
        number_steps += 1

        # add action, observation_delta and reward
        tup = (observation_delta, action_dict[action], reward)
        batch_state_action_reward_tuples.append(tup)

        if reward != 0:
            print('Episode no: %d,  Game round finished, Reward: %f, Total Reward: %f' % (episode_number, reward, total_reward))
            round_number += 1
            n_steps = 0

    # exponentially smoothed version of reward
    if smoothed_reward is None:
        smoothed_reward = total_reward
    else:
        smoothed_reward = smoothed_reward * 0.99 + total_reward * 0.01
    print("Total Reward was %f; running mean reward is %f" % (total_reward, smoothed_reward))

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
        states, actions, rewards = zip(*batch_state_action_reward_tuples)
        rewards = discount_rewards(rewards, discount_factor)
        rewards = rewards - np.mean(rewards)
        rewards = rewards / np.std(rewards)
        batch_state_action_reward_tuples = list(zip(states, actions, rewards))
        policyTFNetwork.train(batch_state_action_reward_tuples) 
        batch_state_action_reward_tuples = []

    # save episodes for checkpoint
    if episode_number % checkpointNumber == 0:
        policyTFNetwork.save_checkpoint()

    episode_number += 1


##封装之一，我们不希望反复输入状态，我们希望
#对control的状态进行检查，输入可以是tensor的可以是numpy的甚至可以是list的
#dtype也要进行检查，最终排列为纵向的numpymat提交