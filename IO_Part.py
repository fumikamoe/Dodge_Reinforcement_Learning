# -*- coding: utf-8 -*-
import win32gui
import win32api
import win32con
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

def init():
    window_name = win32gui.FindWindow(None, TARGET_NAME)
    left, top, right, bot, = win32gui.GetWindowRect(window_name)
    w = right - left
    h = bot - top
    gameWindow = {'top': top+45, 'left': left, 'width': w, 'height': h-45}
    return gameWindow

def gamestart(): # Key Press Enter
    win32api.keybd_event(CONTROL["Enter"], 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(CONTROL["Enter"], 0, win32con.KEYEVENTF_KEYUP, 0)

def gamedone(): # Key press N
    win32api.keybd_event(CONTROL['N'], 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(CONTROL['N'], 0, win32con.KEYEVENTF_KEYUP, 0)

def action(input):
    if input == 0:
        win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Left")
        #time.sleep(0.05)

    if input == 1:
        win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Right")
        # time.sleep(0.05)

    if  input == 2:
        win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Up")
        #time.sleep(.005)

    if input == 3:
            win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
            time.sleep(.1)
            win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
            print("Down")
        # time.sleep(.005)

    if input == 4:
            win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
            win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
            time.sleep(.1)
            win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
            print("Up+Left")

    if input == 5:
            win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
            win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
            time.sleep(.1)
            win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
            print("Up+Right")

    if input == 6:
            win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
            win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
            time.sleep(.1)
            win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
            print("Down+Left")

    if input == 7:
            win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
            win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
            time.sleep(.1)
            win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
            print("Down+Right")

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
    except ValueError:
        time.sleep(1)
        score_text = win32gui.GetWindowText(hwnd) #Text를 뽑아낸다
        start = score_text.index('(') # slicing start
        end = score_text.index(')') # Slicing end
        pass
    score_text = score_text[start+2:end-2] #Score부분만 Slicing

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
        gameWindow = init()

        #capture screen
        sct = mss()
        sct.get_pixels(gameWindow)
        img = Image.frombytes('L', (sct.width, sct.height), sct.image)

        #dataset downsample to Black & white
        bw = np.asarray(img).copy()
        bw[bw < 200] = 0
        bw[bw >= 200] = 255
        img = np.array(bw)

        Result_screen = win32gui.FindWindow(None, '닷지')

        if Result_screen: # 결과 화면이 있으면
            gamedone() # N을 눌러 창을 닫는다
            time.sleep(.05)
        #if end

        action(100) #start or skip Game over
        time.sleep(.05)
        action(100) #restart or 그냥 눌러봄
        observation = img  # return observation data
        #if end

    if is_Activating() == False: # is not activated
        while():
            time.sleep(1)
            print("Not Activate window")
        #while end
    #else end

    return observation

def Game_env(action_input):
    if is_Activating:
        #print("<<game_env_start>>")
        #init part
        reward = 0
        done = False
        _ = None
        gameWindow = init()
        #print("<<capture_screen>>")
        #capture screen
        sct = mss()
        sct.get_pixels(gameWindow)

        #bw = Image.frombytes('L', (sct.width, sct.height), sct.image)

        img = Image.frombytes('L', (sct.width, sct.height), sct.image)

        #dataset downsample to Black & white
        bw = np.asarray(img).copy()
        bw[bw < 200] = 0
        bw[bw >= 200] = 255

        img = np.array(bw)

        observation = img #return observation data

        Result_screen = win32gui.FindWindow(None, '닷지')

    if Result_screen == False: #if is not done
        #print("<<Action>>")
        action(action_input)
        #print(img)
        #print(img.shape)
        #print(np.reshape(img, (1, -1)).shape)
        reward = 1


    if Result_screen:
        #스크린 떴을때
        #print("Game Over!")
        #reward = find_score(Result_screen) #메세지 창에서 점수 추출
        done = True


    return observation, reward, done, _


'''
done = False
time.sleep(5)


while 1:
        time.sleep(2)
        reset_env()
        step = 0

        while done == False:
            control_rand = np.random.random_integers(0, 7, None)
            obs, reward, done, _ = Game_env(control_rand)
            step += 1
            print("-----------------------------------")
            print("step is {}".format(step))
            print("action : {}".format(control_rand))
            print("Observation : {}".format(obs))
            print("Reward : {}".format(reward))
            print("done : {}".format(done))
            print("-----------------------------------")
'''