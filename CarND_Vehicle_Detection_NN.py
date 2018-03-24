# written by Mohammad Reza Kakoee 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from all_functions import *
from sklearn.model_selection import GridSearchCV
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
import pickle
#from sklearn.cross_validation import train_test_split




## debug
debug_read_video=1

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
y_start_stop = [350, 700] # Min and max in y to search in slide_window()
xstart=0

model_filename = 'finalized_model_NN_rgb.sav'

##


print("==== loading NN model...")
from keras.models import load_model
model = load_model(model_filename)
print("==== load done")


  
    
ystart = y_start_stop[0]
ystop = y_start_stop[1]
scales=[1,1.5,2]
  

frame=0
heat_zero = []
heat_frames=[0,0,0,0,0]
heat_threshold=3
first_frame=True
frame_history=10


def process_image(myimg):
    
    new_heat = find_cars_NN(myimg, color_space, ystart, ystop, xstart, scales,model, pix_per_cell, cell_per_block, heat_threshold)                      

  
    out_img=draw_hit_map(myimg,ystart,xstart,new_heat)
    
    return out_img
 
    

def process_frame(myimg):
    global frame
    global heat_zero
    global first_frame
    global heat_frames
    if(first_frame):
        img_tosearch = myimg[ystart:ystop,:,:]
        heat_zero = np.zeros_like(img_tosearch[:,:,0]).astype(np.float)
        heat_frames= np.array([heat_zero]*10)
        #[heat_zero,heat_zero,heat_zero,heat_zero,heat_zero,heat_zero,heat_zero,heat_zero,heat_zero,heat_zero]

    
    new_heat = find_cars_NN(myimg, color_space, ystart, ystop, xstart, scales,model, pix_per_cell, cell_per_block, heat_threshold)                      

    
    heat_frames[frame%frame_history] = new_heat
    heat_sum = heat_zero
    #for heat_frame in heat_frames:
    #heat_sum = heat_frames[0] + heat_frames[1] + heat_frames[2] + heat_frames[3] + heat_frames[4] + heat_frames[5] + \
                #heat_frames[6] + heat_frames[7] + heat_frames[8] + heat_frames[9] 
                
    heat_sum = np.sum(heat_frames, axis=0)
    heat_sum = apply_threshold(heat_sum,heat_threshold*10)
    out_img=draw_hit_map(myimg,ystart,xstart,heat_sum)
    #out_img=draw_hit_map(myimg,ystart,xstart,new_heat)
    
    first_frame=False
    frame+=1
    if(frame==frame_history):
        for i in range(0,(int)(frame_history/2)):       
            heat_frames[i] = (heat_frames[i]/4)
        #for i in range((int)(frame_history/2),frame_history):       
        #    heat_frames[i] = (heat_frames[i]/2)
        frame=0
    
    return out_img
    

if(debug_read_video==0):    
    images_test = glob.glob('./test_images/*.jpg')
    import os
    for image_f in images_test:
        first_frame=True
        image = mpimg.imread(image_f)
        res_image= process_image(image)
        #image = image.astype(np.float32)/255
        filename=os.path.basename(image_f)
        filename=filename.split('.')[0]
        plt.imsave("output_images_NN/"+filename+'_w.png',res_image)
        #break

## run on video
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
video_name='project_video'#'project_video' 
#'test_video'
first_frame=True  
if(debug_read_video==1):
    white_output = video_name+'_vehicle_det_NN.mp4'
    clip1 = VideoFileClip(video_name+".mp4")#.subclip(20,23)
    #clip1.save_frame('test_images/car6.png',10)
    #clip1.save_frame('test_images/car7.png',11)
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)



    
