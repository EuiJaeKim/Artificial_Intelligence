#!/usr/bin python3
import tkinter.font as tkf
import tkinter as tk
import numpy as np
import cv2
import RPi.GPIO as GPIO
from time import sleep
import datetime
import multiprocessing
import os
from PIL import Image, ImageTk

# Global Var Area
ActuatorDirs = [7,8]
ActuatorPuls = [12,10]
PointX = [0]
PointY = [0]
StepAngle = 0.9
LowSpeed = 0.0004
FastSpeed = LowSpeed / 2
Speed = FastSpeed
OneCycle = int(360 / StepAngle)
OneMM = 51
X = multiprocessing.Value('i', 0)
Y = multiprocessing.Value('i', 0)
N = multiprocessing.Value('i', 0)
OnColor = '#0000FF'
OffColor = '#14325c'
# GPIO Area
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
for val in ActuatorDirs:
        GPIO.setup(val, GPIO.OUT)
for val in ActuatorPuls:
        GPIO.setup(val, GPIO.OUT)
 # self.output_path = /home/pi/Pictures
# Class Area
class SMTNC001(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self,string = "SMT_Nozzle_Checker")
        tk.Tk.geometry(self,"800x410+0+0")
        tk.Tk.resizable(self,False,False)

        self.output_path = '/home/pi/Saved_Data/'
        self.current_image = None

        # ----- frame -----
        self.left_frame = tk.Frame(self, borderwidth=5, relief=tk.RIDGE, height=450, width=500)
        self.vs = cv2.VideoCapture(0)
        self.width = self.vs.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.size = (300, 450)
        self.lmain = tk.Label(self.left_frame)
        self.lmain.pack()
        self.video_loop()
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        self.right_frame = tk.Frame(self, background="white", borderwidth=5, relief=tk.RIDGE, height=450, width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
        self.Button_init()
    def video_loop(self):
        ok, frame = self.vs.read()  # read frame from video stream
        if N.value == 1:
            print("N.value is 1")
            ts = datetime.datetime.now()
            filename = "Vedio/{}.avi".format(ts.strftime("%Y-%m-%d-%H-%M-%S"))
            p = os.path.join(self.output_path, filename)
            width = self.vs.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
            size = (int(width), int(height))
            self.out = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'DIVX'), 10.0, size)
            N.value == 2
        elif N.value == 3:
            print("N.value is 3")
            self.out.write(frame)
        elif N.value == 4:
            self.out.release()
            N.value = 0
        frame = cv2.resize(frame,(450,300))
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
        self.current_image = Image.fromarray(cv2image)  # convert image for PIL
        imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
        self.lmain.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
        self.lmain.configure(image=imgtk)  # show the image
        self.lmain.after(20, self.video_loop)  # call the same function after 30 milliseconds
    def Button_init(self):
        self.ButtonSize = 45
        self.ButtonFontSize = 13
        self.ButtonList = []

        self.B12P = tk.Button(self.right_frame, text="12P", font = tkf.Font(family="Helvetica", size=20), command=self.Switch)
        self.B12P.configure(background=OffColor)
        self.B24P = tk.Button(self.right_frame, text="24P", font = tkf.Font(family="Helvetica", size=20), command=self.Switch)
        self.B24P.configure(background=OnColor)

        self.B1 = tk.Button(self.right_frame, text="1", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B1_click_event)
        self.ButtonList.append(self.B1)
        self.B2 = tk.Button(self.right_frame, text="2", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B2_click_event)
        self.ButtonList.append(self.B2)
        self.B3 = tk.Button(self.right_frame, text="3", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B3_click_event)
        self.ButtonList.append(self.B3)
        self.B4 = tk.Button(self.right_frame, text="4", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B4_click_event)
        self.ButtonList.append(self.B4)
        self.B5 = tk.Button(self.right_frame, text="5", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B5_click_event)
        self.ButtonList.append(self.B5)
        self.B6 = tk.Button(self.right_frame, text="6", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B6_click_event)
        self.ButtonList.append(self.B6)
        self.B7 = tk.Button(self.right_frame, text="7", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B7_click_event)
        self.ButtonList.append(self.B7)
        self.B8 = tk.Button(self.right_frame, text="8", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B8_click_event)
        self.ButtonList.append(self.B8)
        self.B9 = tk.Button(self.right_frame, text="9", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B9_click_event)
        self.ButtonList.append(self.B9)
        self.B10 = tk.Button(self.right_frame, text="10", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B10_click_event)
        self.ButtonList.append(self.B10)
        self.B11 = tk.Button(self.right_frame, text="11", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B11_click_event)
        self.ButtonList.append(self.B11)
        self.B12 = tk.Button(self.right_frame, text="12", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B12_click_event)
        self.ButtonList.append(self.B12)
        self.B13 = tk.Button(self.right_frame, text="13", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B13_click_event)
        self.ButtonList.append(self.B13)
        self.B14 = tk.Button(self.right_frame, text="14", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B14_click_event)
        self.ButtonList.append(self.B14)
        self.B15 = tk.Button(self.right_frame, text="15", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B15_click_event)
        self.ButtonList.append(self.B15)
        self.B16 = tk.Button(self.right_frame, text="16", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B16_click_event)
        self.ButtonList.append(self.B16)
        self.B17 = tk.Button(self.right_frame, text="17", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B17_click_event)
        self.ButtonList.append(self.B17)
        self.B18 = tk.Button(self.right_frame, text="18", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B18_click_event)
        self.ButtonList.append(self.B18)
        self.B19 = tk.Button(self.right_frame, text="19", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B19_click_event)
        self.ButtonList.append(self.B19)
        self.B20 = tk.Button(self.right_frame, text="20", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B20_click_event)
        self.ButtonList.append(self.B20)
        self.B21 = tk.Button(self.right_frame, text="21", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B21_click_event)
        self.ButtonList.append(self.B21)
        self.B22 = tk.Button(self.right_frame, text="22", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B22_click_event)
        self.ButtonList.append(self.B22)
        self.B23 = tk.Button(self.right_frame, text="23", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B23_click_event)
        self.ButtonList.append(self.B23)
        self.B24 = tk.Button(self.right_frame, text="24", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B24_click_event)
        self.ButtonList.append(self.B24)
        
        self.BAuto = tk.Button(self.right_frame, text="Auto", font = tkf.Font(family="Helvetica", size=20), command=self.BAuto_click_event)
        self.BCapture = tk.Button(self.right_frame, text="Capture", font = tkf.Font(family="Helvetica", size=17), command=self.BCapture_click_event) # take_snapshot
        self.BReset = tk.Button(self.right_frame, text="Reset", font = tkf.Font(family="Helvetica", size=20) , command=self.BReset_click_event)
        self.BReset.configure(background="red")

        self.B12P.pack()
        self.B24P.pack()
        for val in self.ButtonList:
            val.pack()
        self.BAuto.pack()
        self.BCapture.pack()
        self.BReset.pack()
        X = 10
        Y = 70
        for i, val in enumerate(self.ButtonList):
            if i % 6 == 0 :
                X = 10
                Y += self.ButtonSize
            val.place(x=X, y=Y, width=self.ButtonSize, height=self.ButtonSize)
            X += self.ButtonSize
        X = 10
        self.B12P.place(x=X,y=10, width=self.ButtonSize*2, height=self.ButtonSize*2)
        self.B24P.place(x=self.ButtonSize*2+20,y=10, width=self.ButtonSize*2, height=self.ButtonSize*2)
        self.B24P.config(state=tk.DISABLED)
        self.BAuto.place(x=X, y=Y + self.ButtonSize + 10, width=self.ButtonSize*2, height=self.ButtonSize*2)
        self.BCapture.place(x=X+self.ButtonSize*2, y=Y + self.ButtonSize + 10, width=self.ButtonSize*2, height=self.ButtonSize*2)
        self.BReset.place(x=X+self.ButtonSize*4, y=Y + self.ButtonSize + 10, width=self.ButtonSize*2, height=self.ButtonSize*2)
    def B1_click_event(self):
        GoalX = 0
        GoalY = 0
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B2_click_event(self):
        GoalX = 0
        GoalY = 1
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B3_click_event(self):
        GoalX = 0
        GoalY = 2
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B4_click_event(self):
        GoalX = 0
        GoalY = 3
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B5_click_event(self):
        GoalX = 0
        GoalY = 4
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B6_click_event(self):
        GoalX = 0
        GoalY = 5
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B7_click_event(self):
        GoalX = 1
        GoalY = 0
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B8_click_event(self):
        GoalX = 1
        GoalY = 1
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B9_click_event(self):
        GoalX = 1
        GoalY = 2
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B10_click_event(self):
        GoalX = 1
        GoalY = 3
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B11_click_event(self):
        GoalX = 1
        GoalY = 4
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B12_click_event(self):
        GoalX = 1
        GoalY = 5
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B13_click_event(self):
        GoalX = 2
        GoalY = 0
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B14_click_event(self):
        GoalX = 2
        GoalY = 1
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B15_click_event(self):
        GoalX = 2
        GoalY = 2
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B16_click_event(self):
        GoalX = 2
        GoalY = 3
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B17_click_event(self):
        GoalX = 2
        GoalY = 4
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B18_click_event(self):
        GoalX = 2
        GoalY = 5
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B19_click_event(self):
        GoalX = 3
        GoalY = 0
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B20_click_event(self):
        GoalX = 3
        GoalY = 1
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B21_click_event(self):
        GoalX = 3
        GoalY = 2
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B22_click_event(self):
        GoalX = 3
        GoalY = 3
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B23_click_event(self):
        GoalX = 3
        GoalY = 4
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def B24_click_event(self):
        GoalX = 3
        GoalY = 5
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,GoalX))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,GoalY))
        worker_1.start()
        worker_2.start()
    def BAuto_click_event(self):
        # 녹화 시작.
        N.value = 1
        worker_1 = multiprocessing.Process(target=Auto_Move, args=(N,))
        worker_1.start()
    def BCapture_click_event(self):
        ts = datetime.datetime.now() 
        filename = "Picture/{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S")) 
        p = os.path.join(self.output_path, filename)
        self.current_image.save(p, "JPEG")
        print("[INFO] saved {}".format(filename))
    def BReset_click_event(self):
        print("24PBReset")
        pass
    def Switch(self):
        if self.B24P['state'] == 'disabled':
            self.B24P.config(state=tk.NORMAL)
            self.B12P.config(state=tk.DISABLED)
            for _,val in enumerate(self.ButtonList[12:]):
                val.destroy()
            del self.ButtonList[12:]
        else:
            self.B12P.config(state=tk.NORMAL)
            self.B24P.config(state=tk.DISABLED)
            
            self.B13 = tk.Button(self.right_frame, text="13", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B13_click_event)
            self.ButtonList.append(self.B13)
            self.B14 = tk.Button(self.right_frame, text="14", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B14_click_event)
            self.ButtonList.append(self.B14)
            self.B15 = tk.Button(self.right_frame, text="15", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B15_click_event)
            self.ButtonList.append(self.B15)
            self.B16 = tk.Button(self.right_frame, text="16", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B16_click_event)
            self.ButtonList.append(self.B16)
            self.B17 = tk.Button(self.right_frame, text="17", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B17_click_event)
            self.ButtonList.append(self.B17)
            self.B18 = tk.Button(self.right_frame, text="18", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B18_click_event)
            self.ButtonList.append(self.B18)
            self.B19 = tk.Button(self.right_frame, text="19", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B19_click_event)
            self.ButtonList.append(self.B19)
            self.B20 = tk.Button(self.right_frame, text="20", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B20_click_event)
            self.ButtonList.append(self.B20)
            self.B21 = tk.Button(self.right_frame, text="21", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B21_click_event)
            self.ButtonList.append(self.B21)
            self.B22 = tk.Button(self.right_frame, text="22", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B22_click_event)
            self.ButtonList.append(self.B22)
            self.B23 = tk.Button(self.right_frame, text="23", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B23_click_event)
            self.ButtonList.append(self.B23)
            self.B24 = tk.Button(self.right_frame, text="24", font = tkf.Font(family="Helvetica", size=self.ButtonFontSize), command=self.B24_click_event)
            self.ButtonList.append(self.B24)

            for _,val in enumerate(self.ButtonList[12:]):
                val.pack()

            X = 10
            Y = 70 + (self.ButtonSize*2)
            for i, val in enumerate(self.ButtonList[12:]):
                if i % 6 == 0 :
                    X = 10
                    Y += self.ButtonSize
                val.place(x=X, y=Y, width=self.ButtonSize, height=self.ButtonSize)
                X += self.ButtonSize
            
        # self.master.switch_frame(ASM1000)

def Auto_Move(num):
    if X.value != 0 or Y.value != 0:
        worker_1 = multiprocessing.Process(target=MoveX, args=(X,0))
        worker_2 = multiprocessing.Process(target=MoveY, args=(Y,0))
        worker_1.start()
        worker_2.start()

        worker_1.join()
        worker_2.join()
    if N.value != 2 :
         sleep(0.5)
    num.value = 3
    sleep(1)
    
    for i in range(4):
        x = i
        for j in range(6):
            y = j
            worker_1 = multiprocessing.Process(target=MoveX, args=(X,x))
            worker_2 = multiprocessing.Process(target=MoveY, args=(Y,y))
            worker_1.start()
            worker_2.start()

            worker_1.join()
            worker_2.join()
            sleep(1)
    num.value = 4
    sleep(1)
    worker_1 = multiprocessing.Process(target=MoveX, args=(X,0))
    worker_2 = multiprocessing.Process(target=MoveY, args=(Y,0))
    worker_1.start()
    worker_2.start()

    worker_1.join()
    worker_2.join()
def MoveX(X,GoalX):
    DirX = X.value - PointX[GoalX]
    X.value = PointX[GoalX]
    if DirX < 0:
        GPIO.output(ActuatorDirs[0], GPIO.LOW)
        for i in range (int(abs(DirX))):
            GPIO.output(ActuatorPuls[0], GPIO.HIGH)
            sleep(Speed)
            GPIO.output(ActuatorPuls[0], GPIO.LOW)
            sleep(Speed)
    elif DirX > 0:
        GPIO.output(ActuatorDirs[0], GPIO.HIGH)
        for i in range (int(abs(DirX))):
            GPIO.output(ActuatorPuls[0], GPIO.HIGH)
            sleep(Speed)
            GPIO.output(ActuatorPuls[0], GPIO.LOW)
            sleep(Speed)
def MoveY(Y,GoalY):
    DirY = Y.value - PointY[GoalY]
    Y.value = PointY[GoalY]
    if DirY < 0:
        GPIO.output(ActuatorDirs[1], GPIO.LOW)
        for i in range (int(abs(DirY))):
            GPIO.output(ActuatorPuls[1], GPIO.HIGH)
            sleep(Speed)
            GPIO.output(ActuatorPuls[1], GPIO.LOW)
            sleep(Speed)
    elif DirY > 0:
        GPIO.output(ActuatorDirs[1], GPIO.HIGH)
        for i in range (int(abs(DirY))):
            GPIO.output(ActuatorPuls[1], GPIO.HIGH)
            sleep(Speed)
            GPIO.output(ActuatorPuls[1], GPIO.LOW)
            sleep(Speed)

if __name__ == "__main__":
    for i in range(1,4):
        PointX.append(PointX[i-1]+2180)
    for i in range(1,6):
        PointY.append(PointY[i-1]+1495)
    app = SMTNC001()
    app.mainloop()