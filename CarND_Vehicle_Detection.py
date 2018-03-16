import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from all_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
import pickle
#from sklearn.cross_validation import train_test_split

## debug
debug_train=0
debug_save_model =1
debug_read_video=1
##



#1st Training

# Read in cars and notcars
images_notcars = glob.glob('./training/non-vehicles/**/*.png')
images_cars_far = glob.glob('./training/vehicles/GTI_Far/*.png')
images_cars_Left = glob.glob('./training/vehicles/GTI_Left/*.png')
images_cars_Right = glob.glob('./training/vehicles/GTI_Right/*.png')
images_cars_MiddleClose = glob.glob('./training/vehicles/GTI_MiddleClose/*.png')
images_cars_KITTI = glob.glob('./training/vehicles/KITTI_extracted/*.png')



cars = []
notcars = []
for image in images_notcars:
    notcars.append(image)
    
for image in images_cars_far:
    cars.append(image)
for image in images_cars_Left:
    cars.append(image)
for image in images_cars_Right:
    cars.append(image)
for image in images_cars_MiddleClose:
    cars.append(image)
for image in images_cars_KITTI:
    cars.append(image)

np.random.shuffle(cars) 
np.random.shuffle(notcars)

cars=cars[0:(int)(len(cars)/2)]
 
 
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [380, 660] # Min and max in y to search in slide_window()
xstart=0

model_filename = 'finalized_model.sav'
scaler_filename = 'finalized_scaler.std'


if(debug_train==1):
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    if(debug_save_model==1):
        pickle.dump(svc, open(model_filename, 'wb'))  
        pickle.dump(X_scaler, open(scaler_filename, 'wb'))         
else:
    print("loading SVM model and standardScaler...")
    svc = pickle.load(open(model_filename, 'rb'))
    X_scaler = pickle.load(open(scaler_filename, 'rb'))
    print("load done")

    


    
ystart = y_start_stop[0]
ystop = y_start_stop[1]
scales=[1,1.5,2]
  

frame=0
heat_zero = []
heat_frames=[0,0,0,0,0]
heat_threshold=12
first_frame=True
 
    
def process_frame_old(myimg):

    draw_image = np.copy(myimg)
    windows = slide_window(myimg, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.7, 0.7))

    hot_windows = search_windows(myimg, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    return window_img

def process_frame(myimg):
    global frame
    global heat_zero
    global first_frame
    global heat_frames
    if(first_frame):
        img_tosearch = myimg[ystart:ystop,:,:]
        heat_zero = np.zeros_like(img_tosearch[:,:,0]).astype(np.float)
        heat_frames= [heat_zero,heat_zero,heat_zero,heat_zero,heat_zero]

    
    new_heat = find_cars(myimg, color_space, ystart, ystop, xstart,scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,heat_threshold,heat_zero)                      

    
    heat_frames[frame%5] = new_heat
    heat_sum = heat_zero
    for heat_frame in heat_frames:
        heat_sum += heat_frame 
    heat_sum = apply_threshold(heat_sum,heat_threshold*4)
    out_img=draw_hit_map(myimg,ystart,xstart,heat_sum)
    
    first_frame=False
    frame+=1
    return out_img
    
    
    
image = mpimg.imread('test_images/test6.jpg')

res_image= process_frame(image)


#image = image.astype(np.float32)/255

plt.imsave("output_images/test6_window.jpg",res_image)

## run on video
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
video_name='project_video' 
#'test_video'
first_frame=True  
if(debug_read_video==1):
    white_output = video_name+'_vehicle_det.mp4'
    clip1 = VideoFileClip(video_name+".mp4")#.subclip(0,5)
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)



    
