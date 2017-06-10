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

import pickle
import matplotlib.pyplot as plt
import numpy as np

f = open('./progress_result.txt','rb')
a = pickle.load(f)
print(a)
print(len(a))
f.close()
queue_size = []
queue2 = []
for i in range(len(a)):
    queue_size.append(i)

#print(a)

plt.autoscale(enable=True, axis=u'both', tight=False)
plt.xlabel("EPISODE")
plt.ylabel("SCORE")
plt.xticks(np.arange(0, len(a) + 1, 10))
plt.plot(queue_size, a)
#plt.scatter(queue_size, a)
plt.show()

def init():
    a = [0]
    f = open('./Progress_result.txt', 'wb')
    pickle.dump(a, f)
    f.close()
    print("==이태까지의 진행 과정을 저장했습니다!==")
init()
'''
import win32gui
import win32api
import win32con
import numpy as np
import cv2
from mss import mss
from PIL import Image
TARGET_NAME = '닷지 1.9'

mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}

sct = mss()
def init():
    window_name = win32gui.FindWindow(None, TARGET_NAME)
    left, top, right, bot, = win32gui.GetWindowRect(window_name)
    w = right - left
    h = bot - top
    mon = {'top': top + 47, 'left': left + 2, 'width': w - 4, 'height': h - 50}

    top = top + 47
    left = left + 2
    width = w - 4
    height = h - 50

    return mon
    #return left,top,width,height

while 1:
    #frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    sct = mss()
    sct.get_pixels(init())
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img_np = np.array(img)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    bw = np.asarray(img_np).copy()
    #bw[bw == 255] = 0
    bw[bw < 200] = 0
    bw[bw >= 200] = 255
    img_np = bw

    cv2.imshow('test', img_np)
    print(np.reshape(img_np, (1, -1)).shape)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
'''