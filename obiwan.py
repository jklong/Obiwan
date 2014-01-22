import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt

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

    return (pts[0], (pts[1] + vidHeight) % vidHeight)

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


running = False
roi_size = 25
roi_x = -1
roi_y  = -1
roi = None

if len(sys.argv) != 2:
    sys.exit('Usage: obiwan.py [video file]')

#Try to open video
cap = cv2.VideoCapture(sys.argv[1])

if not cap.isOpened():
    sys.exit('Error opening video file')

#Get video properties
vidWidth = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
vidHeight = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
vidFPS = cap.get(cv2.cv.CV_CAP_PROP_FPS)
vidLen = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
startFrame = 0
endFrame = vidLen

#Define windows
cv2.namedWindow('Obiwan')
cv2.setMouseCallback('Obiwan', set_roi)

cv2.namedWindow('obwControls')
cv2.createTrackbar('ROI size', 'obwControls', roi_size, 100, set_roi_size)
cv2.createTrackbar('Start', 'obwControls', 0, int(vidLen), setStartFrame )
cv2.createTrackbar('End','obwControls',int(vidLen),int(vidLen), setEndFrame )

roi_points = []
vel = []
acc = []
seconds = []
#if true the graph will display and update in real time. Slows down execution
ANIMATE_GRAPH = False

ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Obiwan',img)
plt.ion()

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

roi_points.append((roi_x,roi_y))
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
    roi_points.append((roi_x,roi_y))
    x0,y0 = roi_points[-2]
    x1,y1 = roi_points[-1]
    
    #vel.append(math.sqrt((roi_y-y0)**2 + (roi_x-x0)**2))
    #Only taking vertical component - test
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
    if ANIMATE_GRAPH and getVidPosition(cap) % 30 == 0:
        plt.plot(seconds, acc, 'r-', seconds, vel, 'b-', linewidth=1)
        plt.ylabel('Force')
        plt.draw()
         
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
plt.plot(seconds, acc, 'r-', seconds, vel, 'b-', linewidth=1)
plt.ylabel('Force')
plt.ioff()
plt.show()
