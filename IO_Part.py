# -*- coding: utf-8 -*-
import win32gui
import win32api
import win32con
import os
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
    start_time = datetime.datetime.now()
    return start_time

def gamedone(start_time):
    time_buffer2 = datetime.datetime.now()
    #print(time_buffer2 - start_time)
    win32api.keybd_event(0x4E, 0, 0, 0)
    time.sleep(.05)
    win32api.keybd_event(0x4E, 0, win32con.KEYEVENTF_KEYUP, 0)

def init_game_status():
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

while 1:
    gameWindow = init_game_status()
    sct = mss()
    sct.get_pixels(gameWindow)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    img = Convert_BW(img,200)
    img = np.array(img)


    if win32gui.FindWindow(None,'닷지') == False:
        cv2.imshow('test', img)
        #print(img)
        print(img.shape)
        print(np.reshape(img, (1, -1)).shape)

    if win32gui.FindWindow(None,'닷지'):
        print("Game Over!")
        gamedone(start_time=None)
        time.sleep(.05)
        win32api.keybd_event(0x0D, 0, 0, 0)
        time.sleep(.05)
        win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)

        gamestart()

        #cv2.destroyAllWindows()
        #break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    control_rand = np.random.random_integers(0, 3, None)
    #
    # action(control_rand)