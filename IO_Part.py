# -*- coding: utf-8 -*-
import win32gui
import win32api
import win32con
import cv2
import numpy as np
from mss import mss
from PIL import Image
import time

TARGET_NAME = '닷지 1.9'
RESULT_WINDOW = '닷지'

CONTROL = {'Left':0x25,
           'Up':0x26,
           'Right':0x27,
           'Down':0x28,
           'Enter':0x0D,
            'N':0x4E
}

#axis = [0,0]
kernel = np.ones((3, 3), np.uint8)

def init():
    window_name = win32gui.FindWindow(None, TARGET_NAME)
    left, top, right, bot, = win32gui.GetWindowRect(window_name)
    w = right - left
    h = bot - top
    gameWindow = {'top': top + 47, 'left': left + 2, 'width': w - 4, 'height': h - 50}
    return gameWindow

def gamestart(): # Key Press Enter
    win32api.keybd_event(CONTROL["Enter"], 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(CONTROL["Enter"], 0, win32con.KEYEVENTF_KEYUP, 0)

def gamedone(): # Key press N
    win32api.keybd_event(CONTROL['N'], 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(CONTROL['N'], 0, win32con.KEYEVENTF_KEYUP, 0)

def action(input,axis):
    if input == 0:
        win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Left")
        if not axis[0] <= -18:
            axis[0] -= 1

    if input == 1:
        win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Right")
        if not axis[0] >= 18:
            axis[0] += 1

    if  input == 2:
        win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Up")
        if not axis[1] >= 13:
            axis[1] += 1

    if input == 3:
        win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Down")
        if not axis[1] <= -13:
            axis[1] -= 1

    if input == 4:
        win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
        win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Up+Left")
        axis[0] -= 1
        axis[1] += 1
    if input == 5:
        win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
        win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Up+Right")
        axis[0] += 1
        axis[1] += 1
    if input == 6:
        win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
        win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Down+Left")
        axis[0] -= 1
        axis[1] -= 1
    if input == 7:
        win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
        win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
        # print("Down+Right")
        axis[0] += 1
        axis[1] -= 1
    #arrow key end
    if input == 100:
        win32api.keybd_event(CONTROL["Enter"], 0, 0, 0)
        time.sleep(.05)
        win32api.keybd_event(CONTROL["Enter"], 0, win32con.KEYEVENTF_KEYUP, 0)
#def end

def find_score(hwnd):
    hwnd = win32gui.FindWindowEx(hwnd, 0, "static", None)
    time.sleep(.05)
    try:
        score_text = win32gui.GetWindowText(hwnd) #Text를 뽑아낸다
        start = score_text.index('(') # slicing start
        end = score_text.index(')') # Slicing end
        score_text = score_text[start + 2:end - 2]  # Score부분만 Slicing
    except:
        score_text = 0
        pass
    return float(score_text) # float 값으로 리턴한다
#def end

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 255
    vector[-pad_width[1]:] = 255
    return vector

def reset_env(axis): # 환경 초기화
    Result_screen = win32gui.FindWindow(None, RESULT_WINDOW)
    time.sleep(.1)
    gamedone()  # N을 눌러 창을 닫는다
    time.sleep(.05)

    if Result_screen:
        action(1, axis)
        time.sleep(.05)
        action(100, axis)
    #if end

    action(100, axis)  # start or skip Game over
    time.sleep(.05)
    action(100, axis)  # restart or 그냥 눌러봄

    # first screenshot
    gameWindow = init()
    # capture screen
    sct = mss()

    sct.get_pixels(gameWindow)
    img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img_np = np.array(img_init)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # dataset downsample to Black & white
    bw = np.asarray(img_np).copy()
    bw[bw < 200] = 0  # background
    bw[bw >= 200] = 255  # bullet

    bw[bw == 0] = 0
    bw[bw == 255] = 50
    observation = bw

    for i in range(3):
        sct.get_pixels(gameWindow)
        img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        img_np = np.array(img_init)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # dataset downsample to Black & white
        bw = np.asarray(img_np).copy()
        bw[bw < 200] = 0  # background
        bw[bw >= 200] = 255  # bullet

        bw[bw == 0] = 0
        bw[bw == 255] = 50*i
        observation += bw

    observation = np.lib.pad(observation, 1, padwithtens)
    observation = np.append(axis, observation)
    observation = np.reshape(observation, (-1,)) #reforming ndarray
    return observation, axis

def Game_env(action_input,axis):
    # init part
    gameWindow = init()
    Result_screen = win32gui.FindWindow(None, RESULT_WINDOW)
    # capture screen
    #first screenshot
    sct = mss()
    sct.get_pixels(gameWindow)
    img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img_np = np.array(img_init)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    bw = np.asarray(img_np).copy()
    bw[bw < 200] = 0  # background
    bw[bw >= 200] = 255  # bullet

    bw[bw == 0] = 10
    bw[bw == 255] = 30
    observation = bw

    if Result_screen:
        #결과 하면 검출 시
        time.sleep(0.05)
        living_time = find_score(Result_screen)  # 메세지 창에서 점수 추출
        reward = 0
        done = True
        observation = np.zeros_like(observation)

    if Result_screen == False:  # if is not done
        action(action_input, axis)
        reward = 1
        living_time = 0
        done = False
        for i in range(3):
            sct.get_pixels(gameWindow)
            img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            img_np = np.array(img_init)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # dataset downsample to Black & white
            bw = np.asarray(img_np).copy()
            bw[bw < 200] = 0  # background
            bw[bw >= 200] = 255  # bullet

            bw[bw == 0] = 10
            bw[bw == 255] = 30*(i+1)
            observation += bw

    observation = np.lib.pad(observation, 1, padwithtens) # add to Padding
    observation = np.append(axis,observation)
    observation = np.reshape(observation, (-1,)) #reforming ndarray
    return observation, reward, done, living_time, axis


'''
time.sleep(2)

while 1:
        done = False
        time.sleep(2)
        observation,axis = reset_env()
        step = 0
        while not done:
            control_rand = np.random.random_integers(0, 7, None)
            observation, reward, done, _, _  = Game_env(control_rand,axis)
            step += 1
            print("-----------------------------------")
            print("step is {}".format(step))
            print("action : {}".format(control_rand))
            print("Observation : {}".format(observation))
            print("Reward : {}".format(reward))
            print("done : {}".format(done))
            print(observation.shape)
            print("-----------------------------------")
            cv2.imshow('obs1', observation)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

'''