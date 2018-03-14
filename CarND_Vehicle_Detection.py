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
#from sklearn.cross_validation import train_test_split


x_png=mpimg.imread("./training/non-vehicles/GTI/image2.png")
print("png",np.min(x_png),np.max(x_png))
hsv_png = cv2.cvtColor(x_png, cv2.COLOR_RGB2HSV)
print("png_cv2",np.min(hsv_png),np.max(hsv_png))

jpg = mpimg.imread('./test_images/test6.jpg')
print("jpg",np.min(jpg),np.max(jpg))
hsv_jpg = cv2.cvtColor(jpg, cv2.COLOR_RGB2HSV)
print("jpg_cv2",np.min(hsv_jpg),np.max(hsv_jpg))




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

print(len(cars),len(notcars))
    
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 660] # Min and max in y to search in slide_window()

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


def process_frame(myimg,cspace):

    if(cspace in {'RGB','YCrCb'}):
        myimg=myimg.astype(np.float32)/255

    draw_image = np.copy(myimg)
    windows = slide_window(myimg, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(myimg, windows, svc, X_scaler, color_space=cspace, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    return window_img


image = mpimg.imread('test_images/test6.jpg')



image = image.astype(np.float32)/255

res_image= process_frame(image,color_space)


# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255


plt.imsave("output_images/test6_window.jpg",res_image)
    
