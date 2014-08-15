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

#Not my code, stolen from find_obj
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs
	
# Mostly not my code, stolen from find_obj
#
# Edited to draw centroid on img2
def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
	
	

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
			
	points = [kp[1].pt for kp in kp_pairs]
	centroid = np.mean(points, axis = 0) #Calculate centroid coordinates
		
	cv2.circle(vis, (int(centroid[0] + w1),int(centroid[1])), 2, (0,0,255), -1)

    cv2.imshow(win, vis)
	
running = False
roi_x = -1
roi_y  = -1
roi = None
smooth_len = 11 #Length of smoothing kernal

ap = argparse.ArgumentParser()
ap.add_argument('video', help='The video file to analyse')
ap.add_argument('weight',help='Weight being lifted, no units',type=float)
ap.add_argument('-a', help='Frame number to start analysis',type=int,default=0)
ap.add_argument('-b', help='Frame number to end analysis',type=int, default=-1)
ap.add_argument('-s', help='Hessian threshold for SURF, lower is more sensitive', type=int, default=200)
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

roi_size = int(vidWidth/25.6) #Initial roi size setting. Set based on wild guess and testing.
searchSize = int(roi_size * 1.5) #Size of area to search for template, tweak this for performance and accuracy

#Define windows
cv2.namedWindow('Obiwan')
cv2.setMouseCallback('Obiwan', set_roi)

cv2.namedWindow('obwControls')
cv2.createTrackbar('ROI size', 'obwControls', roi_size, 100, set_roi_size)
cv2.createTrackbar('Start', 'obwControls', startFrame, int(vidLen), setStartFrame )
cv2.createTrackbar('End','obwControls',endFrame,int(vidLen), setEndFrame )

roi_points = []
path = []
vel = []
acc = []
seconds = []


set_vid_position(cap, startFrame)
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Obiwan',img)

while (roi_x == -1 or roi_y == -1):
    cv2.waitKey(10)
	
minHessian = args.s #Default from tutorial is 400
surf = cv2.SURF(minHessian)

roi_kp, roi_des = surf.detectAndCompute(roi,None)
print "%s keypoints from ROI" % len(roi_kp)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)


#cv2.imshow('Obiwan',img)


if getVidPosition(cap) != startFrame:
    # If the last frame we saw before starting was an end frame, reset to start point
    set_vid_position(cap,startFrame)

running = True

while(1):
	ret, img = cap.read()

	if img is None or cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == endFrame:
		break

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kp, des = surf.detectAndCompute(img,None)

	matches = flann.knnMatch(roi_des,des,k=2)

	p1,p2, kp_pairs = filter_matches(roi_kp,kp,matches)
	
	#Skip frames that don't have matches, it crashes our visualisation
	if len(kp_pairs) > 1:
		explore_match('matches',roi,img,kp_pairs)
	else:
		print "Frame %s skipped" % getVidPosition(cap)
		 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cv2.waitKey()

# cap.release()
# cv2.destroyAllWindows()

# plt.subplot(211)
# plt.plot(seconds, map(lambda x: x*args.weight , smooth(np.array(acc),window_len=smooth_len).tolist()), 'r-',linewidth=1)
# plt.ylabel('Force')
# plt.axhline(linewidth=0.75,color='k',ls='--', alpha=0.5 )

# plt.subplot(212)
# plt.plot(seconds, smooth(np.array(vel),window_len=smooth_len), 'b-', linewidth=1)
# plt.ylabel('Bar Speed')
# plt.axhline(linewidth=0.75,color='k',ls='--', alpha=0.5 )
# plt.show()
