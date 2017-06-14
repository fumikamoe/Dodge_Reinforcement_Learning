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

CONTROL = {'Left':0x25,
           'Up':0x26,
           'Right':0x27,
           'Down':0x28,
           'Enter':0x0D,
            'N':0x4E
}

axis = [0,0]

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
        axis[0] -= 1

    if input == 1:
        win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Right")
        axis[0] += 1

    if  input == 2:
        win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Up")
        axis[1] += 1

    if input == 3:
        win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
        #print("Down")
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
            #print("Down+Right")
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

def is_Activating():
    active_window = win32gui.GetForegroundWindow()
    active_window = win32gui.GetWindowText(active_window)

    if active_window == TARGET_NAME or active_window == "닷지":
        active_status = True
    else :
        active_status = False
        print("Game is not Activated!!!!")
    return active_status
#def end

def reset_env(): # 환경 초기화
    if is_Activating(): # 창이 활성화 중일 떄
        Result_screen = win32gui.FindWindow(None, '닷지')

        if Result_screen: # 결과 화면이 있으면
            gamedone() # N을 눌러 창을 닫는다
            time.sleep(.05)
        #if end

        action(100,axis) #start or skip Game over
        time.sleep(.05)
        action(100,axis) #restart or 그냥 눌러봄

        gameWindow = init()
        # capture screen
        sct = mss()
        sct.get_pixels(gameWindow)
        img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        img_np = np.array(img_init)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # dataset downsample to Black & white
        bw = np.asarray(img_np).copy()
        bw[bw < 200] = 0
        bw[bw >= 200] = 255

        obs1 = bw  # return observation data
        '''
        sct = mss()
        sct.get_pixels(gameWindow)
        img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        img_np = np.array(img_init)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # dataset downsample to Black & white
        bw = np.asarray(img_np).copy()
        bw[bw < 200] = 0
        bw[bw >= 200] = 255
        '''
        obs2 = bw
    #if end
    if is_Activating() == False: # is not activated
        time.sleep(1)
        print("Not Activate window")
        #while end
    #lsee end
    return obs1,obs2

def Game_env(action_input,axis):
    if is_Activating:
        #init part
        reward = 0
        done = False
        _ = None
        gameWindow = init()
        #capture screen

        sct = mss()
        sct.get_pixels(gameWindow)
        img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        img_np = np.array(img_init)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        bw = np.asarray(img_np).copy()
        bw[bw < 200] = 0
        bw[bw >= 200] = 255
        obs1 = bw

        Result_screen = win32gui.FindWindow(None, '닷지')

    if Result_screen == False: #if is not done
        action(action_input,axis)
        if axis[0] <= -16 or axis[0] >= 16 or axis[1] <= -12 or axis[1] >= 12:
            reward = -1.0
            living_time = 0
            done = True
            print("Warning!")
            time.sleep(5)
        else:
            reward = 1.0
            living_time = 0

    if Result_screen:
        #스크린 떴을때
        living_time = find_score(Result_screen) #메세지 창에서 점수 추출
        done = True
    '''
    sct.get_pixels(gameWindow)
    img_init = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img_np = np.array(img_init)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # dataset downsample to Black & white
    bw = np.asarray(img_np).copy()
    bw[bw < 200] = 0
    bw[bw >= 200] = 255
    '''
    obs2 = bw
    return obs1, obs2, reward, done, living_time, axis

'''

time.sleep(2)

while 1:
        done = False
        time.sleep(2)
        obs1, obs2 = reset_env()
        step = 0
        while not done:
            control_rand = np.random.random_integers(0, 7, None)
            obs1, obs2, reward, done, _, _  = Game_env(control_rand,axis)
            step += 1
            print("-----------------------------------")
            print("step is {}".format(step))
            print("action : {}".format(control_rand))
            print("Observation : {}".format(obs1))
            print("Reward : {}".format(reward))
            print("done : {}".format(done))
            #print(np.reshape(obs1, (1, -1)).shape)
            print(obs1.shape)
            print(np.size(obs1))
            print("size is : {} x {}".format(np.size(obs1,1),np.size(obs1,0)))
            print("-----------------------------------")
'''