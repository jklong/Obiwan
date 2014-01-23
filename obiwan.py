import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
import argparse

def get_roi_corners():
    
    roi_x1 = roi_x - roi_size
    roi_y1 = roi_y - roi_size
    roi_x2 = roi_x + roi_size
    roi_y2 = roi_y + roi_size
    
    return (roi_x1, roi_y1, roi_x2, roi_y2)

def set_roi( event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        global roi_x
        global roi_y
        global roi 

        roi_x = x
        roi_y = y

        x1,y1,x2,y2 = get_roi_corners()

        roi = img[y1:y2, x1:x2]
        print '#ROI set at t= '+ str(getVidPosition(cap)/vidFPS)
    
def set_roi_size( r):
    global roi_size
    roi_size = r

def draw_roi():
    x1,y1,x2,y2 = get_roi_corners()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.circle(img, (roi_x, roi_y), 2, (255, 0, 0), 1)

def convScreenToCart(pts):
    
    #Screen coordinates have (0,0) at top left but cartesian has bottom left

    return (pts[0], -1 * pts[1] + vidHeight)

def set_vid_position(cap, pos):
    #Set position at frame pos
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)

def getVidPosition(cap):
    return cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

def setStartFrame(n):
    global startFrame
    global img

    if not running and not cap is None:
        set_vid_position(cap, n)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Obiwan',img)
        startFrame = n

def setEndFrame(n):
    global endFrame
    global img

    if not running and not cap is None:
        set_vid_position(cap, n)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Obiwan',img)
        endFrame = n

#Stolen from scipy recipe
def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimensional arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size"
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming','bartlett','blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat':
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

running = False
roi_size = 25
roi_x = -1
roi_y  = -1
roi = None
smooth_len = 11 #Length of smoothing kernal

ap = argparse.ArgumentParser()
ap.add_argument('video', help='The video file to analyse')
ap.add_argument('weight',help='Weight being lifted, no units',type=float)
ap.add_argument('-a', help='Frame number to start analysis',type=int,default=0)
ap.add_argument('-b', help='Frame number to end analysis',type=int, default=-1)
args = ap.parse_args()

#Try to open video
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    sys.exit('Error opening video file')

#Get video properties
vidWidth = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
vidHeight = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
vidFPS = cap.get(cv2.cv.CV_CAP_PROP_FPS)
vidLen = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
startFrame = args.a
endFrame = args.b if args.b != -1 else int(vidLen)

#Define windows
cv2.namedWindow('Obiwan')
cv2.setMouseCallback('Obiwan', set_roi)

cv2.namedWindow('obwControls')
cv2.createTrackbar('ROI size', 'obwControls', roi_size, 100, set_roi_size)
cv2.createTrackbar('Start', 'obwControls', startFrame, int(vidLen), setStartFrame )
cv2.createTrackbar('End','obwControls',endFrame,int(vidLen), setEndFrame )

roi_points = []
vel = []
acc = []
seconds = []


set_vid_position(cap, startFrame)
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Obiwan',img)

#Wait for click to define roi
while (roi_x == -1 or roi_y == -1):
    cv2.waitKey(10)

if getVidPosition(cap) != startFrame:
    #If the last frame we saw before starting was an end frame, reset to start point
    set_vid_position(cap,startFrame)
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, roi, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
   
    roi_x, roi_y = max_loc
    roi_x += roi_size 
    roi_y += roi_size

roi_points.append(convScreenToCart((roi_x,roi_y)))
draw_roi()
running = True

while(1):
    ret, img = cap.read()

    if img is None or cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == endFrame:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, roi, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
   
    roi_x, roi_y = max_loc
    roi_x += roi_size 
    roi_y += roi_size

    #Calculate velocity
    roi_points.append(convScreenToCart((roi_x,roi_y)))
    x0,y0 = roi_points[-2]
    x1,y1 = roi_points[-1]
    
    #Only taking vertical component
    vel.append(y1-y0)

    #Calculate acceleration
    if len(vel) == 1:
        acc.append(vel[-1])
    else:
        acc.append(vel[-1] - vel[-2])

    seconds.append(getVidPosition(cap)/vidFPS) 

    draw_roi()

    cv2.putText(img, str(round(seconds[-1],3)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,255 )

    cv2.imshow('Obiwan',img)
         
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

plt.subplot(211)
plt.plot(seconds, map(lambda x: x*args.weight , smooth(np.array(acc),window_len=smooth_len).tolist()), 'r-',linewidth=1)
plt.ylabel('Force')

plt.subplot(212)
plt.plot(seconds, smooth(np.array(vel),window_len=smooth_len), 'b-', linewidth=1)
plt.ylabel('Bar Speed')
plt.show()
