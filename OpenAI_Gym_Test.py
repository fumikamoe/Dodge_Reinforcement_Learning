# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import win32gui
import win32api
import win32con
import numpy as np
import cv2
from mss import mss
from PIL import Image
TARGET_NAME = '닷지 1.9'


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
plt.xticks(np.arange(0, len(a) + 1, 100))
plt.plot(queue_size, a)
#plt.scatter(queue_size, a)
plt.show()

def init():
    a = [0]
    f = open('./Progress_result.txt', 'wb')
    pickle.dump(a, f)
    f.close()
    print("==이태까지의 진행 과정을 저장했습니다!==")
#init()





def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 255
    vector[-pad_width[1]:] = 255
    return vector
kernel = np.ones((3, 3), np.uint8)

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
    bw[bw < 200] = 0
    bw[bw >= 200] = 255
    obs1 = bw
    obs2 = bw
    observation = bw
    print("phase1")

    for i in range(3):
        sct.get_pixels(init())
        img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        img_np = np.array(img_init)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # dataset downsample to Black & white
        bw = np.asarray(img_np).copy()
        bw[bw < 200] = 0  # background
        bw[bw >= 200] = 50*i  # bullet
        obs2 = bw
        observation += obs2
        print("phase2")

    #observation = np.lib.pad(observation, 5, padwithtens)
    cv2.imshow('test', observation)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



a = np.arange(6)
print(a)
a = a.reshape((2, 3))
print(a)

a = np.lib.pad(a, 2, padwithtens)
print(a)
