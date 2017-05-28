# -*- coding: utf-8 -*-
import win32gui
import win32api
import win32con
import re
import numpy as np
import cv2
from mss import mss
from PIL import Image

import time
import datetime


TARGET_NAME = '닷지 1.9'
time1 = 0.0
time2 = 0.0

CONTROL = {'Left':0x25,
           'Up':0x26,
           'Right':0x27,
           'Down':0x28}

def init():
    window_name = win32gui.FindWindow(None, TARGET_NAME)
    left, top, right, bot, = win32gui.GetWindowRect(window_name)
    w = right - left
    h = bot - top
    gameWindow = {'top': top+45, 'left': left, 'width': w, 'height': h-45}
    return gameWindow

def gamestart():
    win32api.keybd_event(0x0D, 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)

def gamedone():
    win32api.keybd_event(0x4E, 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(0x4E, 0, win32con.KEYEVENTF_KEYUP, 0)

def action(input):
    if input == 0:
        win32api.keybd_event(CONTROL["Left"], 0, 0, 0)
        time.sleep(0.1)
        win32api.keybd_event(CONTROL["Left"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Left")
        #time.sleep(0.05)

    if input == 1:
        win32api.keybd_event(CONTROL["Right"], 0, 0, 0)
        time.sleep(0.1)
        win32api.keybd_event(CONTROL["Right"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Right")
        # time.sleep(0.05)

    if  input == 2:
        win32api.keybd_event(CONTROL["Up"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Up"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Up")
        #time.sleep(.005)

    if  input == 3:
        win32api.keybd_event(CONTROL["Down"], 0, 0, 0)
        time.sleep(.1)
        win32api.keybd_event(CONTROL["Down"], 0, win32con.KEYEVENTF_KEYUP, 0)
        print("Down")
        # time.sleep(.005)

    if input == 100:
        win32api.keybd_event(0x0D, 0, 0, 0)
        time.sleep(.05)
        win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)

#def end

def find_score(hwnd):
    hwnd = win32gui.FindWindowEx(hwnd, 0, "static", None)
    score_text = win32gui.GetWindowText(hwnd) #Text를 뽑아낸다
    start = score_text.index('(') # slicing start
    end = score_text.index(')') # Slicing end
    score_text = score_text[start+2:end-2] #Score부분만 Slicing
    return float(score_text) # float 값으로 리턴한다


def Activating():
    active_window = win32gui.GetForegroundWindow()
    active_window = win32gui.GetWindowText(active_window)
    if active_window == TARGET_NAME or active_window == "닷지":
        active_status = True
    else :
        active_status = False
        print("Game is not Activated!!!!")

    return active_status

def reset_env():
    Result_screen = win32gui.FindWindow(None, '닷지')

    if Result_screen:
        gamedone()
        time.sleep(.05)

        action(100)
        time.sleep(.05)
        action(100)

    if Result_screen == False:
        action(100)
        time.sleep(.05)
        action(100)
    done = False
    return done

def Game_env(action_input):
    reward = 0
    done = False
    _ = None
    gameWindow = init()

    sct = mss()
    sct.get_pixels(gameWindow)
    img = Image.frombytes('L', (sct.width, sct.height), sct.image)
    bw = np.asarray(img).copy()
    bw[bw < 200] = 0
    bw[bw >= 200] = 255
    img = np.array(bw)

    observation = img #return observation data

    Result_screen = win32gui.FindWindow(None, '닷지')

    if Result_screen == False:
        action(action_input)
        #print(img)
        #print(img.shape)
        #print(np.reshape(img, (1, -1)).shape)

    if Result_screen:
        print("Game Over!")
        reward = find_score(Result_screen)
        #gamedone()
        time.sleep(.05)
        #win32api.keybd_event(0x0D, 0, 0, 0)
        time.sleep(.05)
        #win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)
        #gamestart()
        done = True


    return observation, reward, done, _

done = False
time.sleep(5)

while 1:
    if Activating():
        time.sleep(2)
        done = reset_env()
        step = 0
        while done == False:
            control_rand = np.random.random_integers(0, 3, None)
            obs, reward, done, _ = Game_env(control_rand)
            step += 1
            print("-----------------------------------")
            print("step is {}".format(step))
            print("action : {}".format(control_rand))
            print("Observation : {}".format(obs))
            print("Reward : {}".format(reward))
            print("done : {}".format(done))
            print("-----------------------------------")