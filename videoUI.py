from tkinter import filedialog as fd, ttk,Toplevel
from utils.nnUtils import sendNN
from utils.nnUtils import _internal
import cv2
import os
import threading

from threading import *
import numpy as np

'''
This file contains the code for tkinter widgets and UI handling for object detection from any video file.
'''

# takes the root window and fits in the other widgets local to context like progress bar, start, stop buttons

def genImageLabel(win):

    # event fires when the file is selected 
    def _selectVideo():
        global vidSelect
        vidSelect = fd.askopenfilename(
            title="Select video file", filetypes=[("mp4 files", ".mp4")]
        )
    # event fires when clicked on the start button. checks for the file and initializes the VideoWriter and video capture instance from the selected file path
    def _start():
        if vidSelect:
            net, classes, layer_names, output_layers = sendNN()
            cap = cv2.VideoCapture(vidSelect)
            _,f1=cap.read()
            out_vid = cv2.VideoWriter(f"annotated_video.mp4",cv2.VideoWriter_fourcc('a','v','c','1'),10,(f1.shape[1],f1.shape[0]))
            
            win.temp=ttk.Progressbar(win,orient='horizontal',mode='indeterminate',length=200)
            win.temp.grid(row=4,column=1)
            win.temp.start()
            # inner function fires when stop is clicked hides the progress bar and other widgets.
            def _stop():
                cap.release()
                win.temp.stop()
                _destroy()
            win.stop=ttk.Button(win,text='Stop',command=_stop)
            win.stop.grid(row=4,column=2)
            bg=threading.Thread(target=_internal,args=(cap,net,output_layers,classes,out_vid))
            bg.start() 
    win.button = ttk.Button(win, text="Select video file", command=_selectVideo)
    win.button.grid(row=3, column=0)
    win.start=ttk.Button(win,text='Start',command=_start)
    win.start.grid(row=4,column=0)
    # custom function to hide the controls from the video selection page. called by the ui.py file.
    def _destroy():
        win.stop.destroy()
        win.temp.destroy()
        win.start.destroy()
        cv2.destroyAllWindows()
        win.button.destroy()
    return _destroy
