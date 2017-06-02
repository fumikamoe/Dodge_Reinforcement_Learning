# -*- coding: utf-8 -*-
'''
import gym
import numpy as np

env = gym.make('Pong-v0')
obs_q = []
act_q = []

for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        print("--------------------------------------------------------")
        #env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print("Obs is : {}".format(observation))
        print("Act is : {}".format(action))
        print("rwd is : {}".format(reward))
        #print(info)
        print(observation.shape)
        act_q.append(action)
        C_obs = np.reshape(observation, (1, -1))
        print(C_obs.size)
        print("--------------------------------------------------------")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
'''

MAX = 256
for i in range(MAX):
        if i == 100 / 4:
            print("학습 25% 완료...")
        if i == 100 / 2:
            print("학습 50% 완료...")
        if i == 100 * 3 / 4:
            print("학습 75% 완료...")
        print(i)