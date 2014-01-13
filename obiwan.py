import numpy as np
import cv2
import sys

class Obiwan:


    def set_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.roi_x = x
            self.roi_y = y
            roi_x1 = self.roi_x - self.roi_size
            roi_y1 = self.roi_y - self.roi_size
            roi_x2 = self.roi_x + self.roi_size
            roi_y2 = self.roi_y + self.roi_size
            self.roi = self.img[roi_y1:roi_y2, roi_x1:roi_x2]
            print 'ROI set at t= '+ str(self.t/self.vidFPS)

    def set_roi_size(self, r):
        self.roi_size = r

    def draw_roi(self):
        roi_x1 = self.roi_x - self.roi_size
        roi_y1 = self.roi_y - self.roi_size
        roi_x2 = self.roi_x + self.roi_size
        roi_y2 = self.roi_y + self.roi_size

        cv2.rectangle(self.img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 1)
        cv2.circle(self.img, (self.roi_x, self.roi_y), 2, (255, 0, 0), 1)

        if self.roi is None:
            self.roi = self.img[roi_y1:roi_y2, roi_x1:roi_x2]

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
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            if self.img is None:
                break

            res = cv2.matchTemplate(self.img, self.roi, cv2.TM_CCOEFF_NORMED)
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
            
            self.roi_x, self.roi_y  = max_loc
            self.roi_x += self.roi_size 
            self.roi_y += self.roi_size

            self.draw_roi()
            cv2.imshow('Obiwan',self.img)
                 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obw = Obiwan(sys.argv)
    obw.go()
