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


def gamestart():
    win32api.keybd_event(0x0D, 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)

def gamedone(start_time):
    win32api.keybd_event(0x4E, 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(0x4E, 0, win32con.KEYEVENTF_KEYUP, 0)

def init_game_window_status():
    window_name = win32gui.FindWindow(None, TARGET_NAME)
    left, top, right, bot, = win32gui.GetWindowRect(window_name)
    w = right - left
    h = bot - top
    gameWindow = {'top': top+45, 'left': left, 'width': w, 'height': h-45}
    return gameWindow #def end

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

#def end

def Convert_BW(img,threhold):
    img = img.convert('L')
    bw = np.asarray(img).copy()
    bw[bw < threhold] = 0
    bw[bw >= threhold] = 255
    return bw

def find_score(hwnd):
    hwnd = win32gui.FindWindowEx(hwnd, 0, "static", None)
    score_text = win32gui.GetWindowText(hwnd) #Text를 뽑아낸다
    start = score_text.index('(') # slicing start
    end = score_text.index(')') # Slicing end
    score_text = score_text[start+2:end-2] #Score부분만 Slicing
    return float(score_text) # float 값으로 리턴한다


def Game_env(action_input):
    reward = 0
    done = False
    _ = None

    gameWindow = init_game_window_status()
    sct = mss()
    sct.get_pixels(gameWindow)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img = Convert_BW(img,200)
    img = np.array(img)
    Result_screen = win32gui.FindWindow(None, '닷지')

    observation = img #return observation data

    if Result_screen == False:
        action(action_input)
        cv2.imshow('test', img)
        #print(img)
        #print(img.shape)
        #print(np.reshape(img, (1, -1)).shape)

    if Result_screen:
        print("Game Over!")
        reward = find_score(Result_screen)
        gamedone(start_time=None)
        time.sleep(.05)
        #win32api.keybd_event(0x0D, 0, 0, 0)
        time.sleep(.05)
        #win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)
        gamestart()
        done = True

    #control_rand = np.random.random_integers(0, 3, None)
    #action(control_rand)
    return observation, reward, done, _