## Project: Search and Sample Return

#### Email:ziyadalkhonein@gmail.com
---



**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./sample.png
[image2]: ./sample2.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (either data you recorded or the test data provided). Add/modify functions to allow for color selection of obstacles and rock samples.

I have run, added and modified functions in the file. My dataset is recorded in Robot Movie.mp4.

i have crearted a new function that use opencv. it takes image and to check the values of HSV of the image. first i needed to transfrom the image from RGB to BGR and then to HSV because the opencv doesn't use RGB. the range of hsv of color lower yellow = [20,100,100] 
and upper yellow [40,255,255]

```
def id_yellow_rock(rock_img):
    bgr = cv2.cvtColor(rock_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

```
to identify obsticle i had to to take the whole vision of the rover and retrun it   

```
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]),M,(img.shape[1],img.shape[0]))
    return warped,mask
```


As is shown in the pic below, the obstacle is red and the navigable area is blue.
https://github.com/ZeadAlkhonein/Robot-Search-and-Sample/blob/master/map.png

#### 2.Populate the process_image() function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap. Run process_image() on your test data using the moviepy functions provided to create video output of your result.
A video is created ./output/test_mapping.mp4

You can see the navigable terrain and obstacles are plot in the worldmap when the rover goes to anywhere.
https://github.com/ZeadAlkhonein/Robot-Search-and-Sample/blob/master/map.png

The process_image() does as follows:

1. Apply perspective tansform on images
```
warped, mask = perspect_transform(img, source, destination)
```
2. With the transformed image perform thresholding for objects of interest: navigable path, obstacles and rocks
the functions are :
```
threshed = color_thresh(warped)
#see if we find some rock
rock_map = id_yellow_rock(warped)

```
3. Once we have the threshold values we convert these to rover coordinates
```
    xpix, ypix = rover_coords(threshed)
```
4. Rover coordinates are then converted to world map coordinates for mapping
```
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]
    world_size = data.worldmap.shape[0]
    scale = 2 * dst_size
    x_world, y_world = pix_to_world(xpix,ypix,xpos,ypos,yaw,world_size, scale)
```
5. Obstacles = RED, Navigable path = blue white = rocks

```
    data.worldmap[y_world, x_world, 2] = 255
    data.worldmap[obs_y_world, obs_x_world, 0] = 255
    data.worldmap[rock_y_world, rock_x_world, :] = 255
```

### Autonomous Navigation and Mapping

You can see the recorded Autonomous Navigation in Robot Movie.mp4


the logic here same as python notebook key things are diffrent is we had to consider that we take image live and it's changing by the second i needed to take rover postion live with every image we take.
Key things needed were a thresholded image of the path, obstacles, and rocks.
we pass these to rover to make it aware of it surroundings. 
we also need to take rover coordinates to make it make decision on where to go and which route is the best


also all the comments on the code.

```
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # i have applied the perspect transform and to the whole vision of the rover and warped
    warped,mask = perspect_transform(Rover.img,source,destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock sample
    threshed = color_thresh(warped)
    
    # apply rocks threashold to identify of rocks
    rock_img = id_yellow_rock(Rover.img)
    # creat a new varible and Calculate the absolute value element-wise and multipy it with mask 
    obs_map = np.absolute(np.float32(threshed)-1) * mask
    
    # Update Rover.vision_image (this will be displayed on left side of screen
    xpix,ypix = rover_coords(threshed)
    # take Rover Postion 
    xpos,ypos = Rover.pos
    # take Rover yaw 
    yaw = Rover.yaw
    
    world_size = Rover.worldmap.shape[0]
    
    scale = 2 * dst_size
    x_world, y_world = pix_to_world(xpix,ypix,xpos,ypos,yaw,world_size, scale)
    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world = pix_to_world(obsxpix,obsypix,xpos,ypos,yaw,world_size,scale)
    # make navigable terrian blue  
    Rover.worldmap[y_world, x_world, 2] = 255
    # make obs terrian red
    Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
    Rover.vision_image = warped
    nav_pix = Rover.worldmap[:,:,2] > 0
    Rover.worldmap[nav_pix, 0] = 0
    # check id there's any rock 
    if rock_img.any():
        # see rock rock coords to rover
        rock_x, rock_y = rover_coords(rock_img)
        rock_dist, rock_angle = to_polar_coords(rock_x, rock_y)
        #Rover.samples_pos = rock_x, rock_y
        #Rover.near_sample = 1
        # take the least number and make it rock postion
        rock_x = np.argmin(rock_x)
        rock_y = np.argmin(rock_y)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw,world_size, scale)
        # if the rover saw rock in the map make it white 
        Rover.worldmap[rock_y_world, rock_x_world, :] = 255
        # the next line make the yellow rock show in Rover Vision Image 
        Rover.vision_image[:,:,1] = Rover.img[:,:,1] * rock_img
    # show the obs in the Rover Vision Image and make it red
    Rover.vision_image[:,:,0] = obs_map *255
    # show the navigable Terrian in the Rover Vision and make it blue 
    Rover.vision_image[:,:,2] = threshed *255
    # Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix, ypix)

```

if the rover sees navigable terrian it will make forward, if it saw obsticale terrian the rover will make 180 and make forward to the navigable terrian. 
if the rover saw rocks the sample counter will ++ 

also all the comments on the code. 

```

import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
             # i have added this line to check if the rover is near a rock if it's added on the samples located counter
            elif Rover.near_sample == 1:
                    Rover.samples_located +=1                     

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.samples_collected +=1
        
    
    return Rover
```


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1- i have tried to send command to the rover to go and pick the yellow rocks but my tries have faild. i hope in the future in my own free time will do it by the end of the course.

2- also i couldn't upload my dataset on github it says the imgs are too large


