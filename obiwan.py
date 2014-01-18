import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt

class Obiwan:

    def get_roi_corners(self):
        
        roi_x1 = self.roi_x - self.roi_size
        roi_y1 = self.roi_y - self.roi_size
        roi_x2 = self.roi_x + self.roi_size
        roi_y2 = self.roi_y + self.roi_size
        
        return (roi_x1, roi_y1, roi_x2, roi_y2)

    def set_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.roi_x = x
            self.roi_y = y

            x1,y1,x2,y2 = self.get_roi_corners()

            self.roi = self.img[y1:y2, x1:x2]
            print '#ROI set at t= '+ str(self.t/self.vidFPS)

    def set_roi_size(self, r):
        self.roi_size = r

    def draw_roi(self):
        x1,y1,x2,y2 = self.get_roi_corners()

        cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(self.img, (self.roi_x, self.roi_y), 2, (255, 0, 0), 1)

    def calc_vel(self, pts):
        #Returns vector of pixel velocity between frames
        #Assume stationary from beginning of film
        vel = [0.0]
        
        # We want magnitude of the velocity only, we don't care about direction
        # v = dr/dt. Assume cartesian distance, dt = 1 (frame)
        for i in range(1, len(pts)-1):
          x0,y0 = pts[i-1]
          x1,y1 = pts[i]
          
          r = math.sqrt((y1-y0)**2 + (x1-x0)**2)
          vel.append(r)

        return vel

    def calc_accel(self,vel):
        accel = []


        # Magnitude of acceleration only, we don't have the direction anyway
        # a = dv/dt, dt = 1 (frame)
        for i in range(1, len(vel)):
            u = vel[i-1]
            v = vel[1]

            a = v-u
            accel.append(a)

        return accel

    def __init__(self, argv):
        
        self.roi_size = 25
        self.roi_x = -1
        self.roi_y  = -1
        self.roi = None

        self.t = 0 #Frame counter
        
        if len(argv) != 2:
            sys.exit('Usage: obiwan.py [video file]')
        
        #Try to open video
        self.cap = cv2.VideoCapture(sys.argv[1])

        if not self.cap.isOpened():
            sys.exit('Error opening video file')
        
        #Get video properties
        self.vidWidth = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.vidHeight = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.vidFPS = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)

        #Define windows
        cv2.namedWindow('Obiwan')
        cv2.setMouseCallback('Obiwan', self.set_roi)

        cv2.namedWindow('obwControls')
        cv2.createTrackbar('ROI size', 'obwControls', self.roi_size, 100, self.set_roi_size)

    def go(self):
        
        roi_points = []


        ret, self.img = self.cap.read()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Obiwan',self.img)
        self.t += 1

        while (self.roi_x == -1 or self.roi_y == -1):
            cv2.waitKey(10)

        self.draw_roi()
        
        while(1):
            self.t += 1
            ret, self.img = self.cap.read()

            if self.img is None:
                break

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(self.img, self.roi, cv2.TM_CCOEFF_NORMED)
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
           
            self.roi_x, self.roi_y = max_loc
            self.roi_x += self.roi_size 
            self.roi_y += self.roi_size
            roi_points.append((self.roi_x, self.roi_y))

            self.draw_roi()
            cv2.imshow('Obiwan',self.img)
                 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(roi_points) > 2:
            roi_vel = self.calc_vel(roi_points)
            accs = self.calc_accel(roi_vel)
            plt.plot(accs)
            plt.ylabel('Force')
            plt.show()
            print max(accs)


        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obw = Obiwan(sys.argv)
    obw.go()
