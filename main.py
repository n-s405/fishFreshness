#Importing Stuff 

#find thresholds for 3 
# #3 dont form a mask 

import cv2
import numpy as np
import copy  
import math
import sys
from matplotlib import pyplot as plt1

raw_frame = cv2.imread('fish01.png')
process_name = 'yosh'   # Variable to save the name of the current file
current_mask = 'yosh'   # Variable to save the name of the current mask 

def sd_calc(data):      # Function to calculate the standard deviation of an array 
    n = len(data)

    if n <= 1:
        return 0.0

    mean, sd = avg_calc(data), 0.01

    # calculate stan. dev.
    for el in data:
        sd += (float(el) - mean)**2
    sd = math.sqrt(sd / float(n-1))

    return sd

def avg_calc(ls):       # Function to calculate the average of an array 
    n, mean = len(ls), 0.0

    if n <= 1:
        return ls[0]

    # calculate average
    for el in ls:
        mean = mean + float(el)
    mean = mean / float(n)

    return mean   

def show_menu():
    #MENU
    print('Fish Freshness Detection v1.2')
    print('MENU')
    print('1 Select a photo')
    print('1.1 Show the selected photo')

    print('2 Create mask')
    print('3 Isolate and save the mask')
    print('4 Calculate SD and Variance')

    print('5 Enter 3 photos for sampling')
    print('5.1 Show Sample Values')
    print('5.2 Compare with Sampled Values')

    print('6 Check fish gill cloudiness')
    print('7 Super Isolation')

def closing_tasks():    #Function to close windows after use 
    cv2.waitKey(0)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    show_menu()

def basic_masking(input_frame):
    blur_frame = cv2.GaussianBlur(input_frame, (3, 3), 0)
    blurImg = cv2.blur(input_frame,(10,10)) 

    #Select blurred or not blurred for thresholding 
    hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    #hsv_frame = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    cv2.imshow(" not blurred",blur_frame)
    cv2.imshow(" blurred",blurImg)
    #Function to find masks for a photo 

    lower_red = np.array([150, 100, 0])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    cv2.imshow("Mask", mask)
   
def isolate_mask(input_frame):
    blur_frame = cv2.GaussianBlur(input_frame, (3, 3), 0)
    blurImg = cv2.blur(input_frame,(10,10)) 

    #Select blurred or not blurred for thresholding 
    hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    #hsv_frame = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    cv2.imshow(" not blurred",blur_frame)
    cv2.imshow(" blurred",blurImg)
    #Function to find masks for a photo 

    lower_red = np.array([150, 100, 0])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    cv2.imshow("Mask", mask)


    #CONTOURING 
    new_frame = input_frame.copy()
    new_frame_2 = input_frame.copy()
    mask2 = np.zeros(new_frame_2.shape[:2],np.uint8)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea)

    for contour in contours:
        area = cv2.contourArea(contour)
 
        if area > 500:  #12 - 500
            cv2.drawContours(new_frame, contour, -1, (0, 255, 0), 3)
            cv2.drawContours(mask2, [contour],-1, 255, -1)
    cv2.imshow("Frame", new_frame)
    
    
    dst = cv2.bitwise_and(new_frame_2, new_frame_2, mask=mask2)
    cv2.imshow("Gill",dst)
    global process_name
    global current_mask

    current_mask = str(process_name) + '_buffer' + '.png'
    cv2.imwrite(current_mask,dst)

def choice5(one1 , two2 , three3):
    avg_main = 0 
    sd_main = 0 
    a_list = list()
    b_list = list()
    c_list = list()

    cv2.imshow('one',one1)
    cv2.imshow('two',two2)
    cv2.imshow('three',three3)
    mask_one1  = return_isolate_mask(one1)
    cv2.imshow('one mask',mask_one1)
    mask_two2  = return_isolate_mask(two2)
    cv2.imshow('two mask',mask_two2)
    mask_three3  = return_isolate_mask(three3)
    cv2.imshow('three mask',mask_three3)

    for i in range(mask_one1.shape[0]):
                for j in range(mask_one1.shape[1]):
                    
                    if mask_one1[i,j,0] == mask_one1[i,j,1]:
                        if mask_one1[i,j,1] == mask_one1[i,j,2]:
                            pass
                    else :
                        a_list.append(mask_one1[i,j,0])

    for i in range(mask_two2.shape[0]):
                for j in range(mask_two2.shape[1]):
                    
                    if mask_two2[i,j,0] == mask_two2[i,j,1]:
                        if mask_two2[i,j,1] == mask_two2[i,j,2]:
                            pass
                    else :
                        b_list.append(mask_two2[i,j,0])

    for i in range(mask_three3.shape[0]):
                for j in range(mask_three3.shape[1]):
                    
                    if mask_three3[i,j,0] == mask_three3[i,j,1]:
                        if mask_three3[i,j,1] == mask_three3[i,j,2]:
                            pass
                    else :
                        c_list.append(mask_one1[i,j,0])

    averages = [avg_calc(a_list),avg_calc(b_list),avg_calc(c_list)]
    standardDeviations = [sd_calc(a_list),sd_calc(b_list),sd_calc(c_list)]

    avg_main = (averages[0]+averages[1]+averages[2])/3
    sd_main = (standardDeviations[0]+standardDeviations[1]+standardDeviations[2])/3
    print('Total Average is : ' + str(avg_main) )
    print('Total Standard Deviation is : ' + str(sd_main) )

    return avg_main , sd_main

def return_isolate_mask(input_frame):

    blur_frame = cv2.GaussianBlur(input_frame, (3, 3), 0)
    hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([150, 100, 0])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    #cv2.imshow("Mask1", mask)
    #CONTOURING 
    new_frame = input_frame.copy()
    new_frame_2 = input_frame.copy()
    mask2 = np.zeros(new_frame_2.shape[:2],np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500:
            cv2.drawContours(new_frame, contour, -1, (0, 255, 0), 3)
            cv2.drawContours(mask2, [contour],-1, 255, -1)
    #cv2.imshow("Frame1", new_frame)       
    dst1 = cv2.bitwise_and(new_frame_2, new_frame_2, mask=mask2)
    #cv2.imshow("Gill1",dst1)
    return dst1

def superisolate(input_frame):


    src = input_frame
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    output_frame = cv2.merge(rgba,4)
    

    return output_frame





def main():

    #Variable Declarations 
    
    show_menu()

    while True:

        #BGR LIST INITIALIZE
        b_list = list()
        g_list = list()
        r_list = list()

        cv2.startWindowThread()
        fishName = 'fish'

        choice = input('Enter Choice\n')
        global raw_frame
        global process_name



        #MENU SELECTION
        if choice == '1' :
            fishNumber = input('Enter Fish Number\n')
            fishName = fishName + str(fishNumber) + '.png' 
            process_name = fishName
            print('You have chosen the fish : ' + fishName)            
            raw_frame = cv2.imread(fishName)
            show_menu()

        if choice == '1.1' :    #View Photo
            cv2.namedWindow('input')  
            cv2.imshow(fishName , raw_frame)
            closing_tasks()
  
        elif choice == '2':     #Find the mask
            cv2.imshow('input',raw_frame)
            basic_masking(raw_frame)
            closing_tasks()
        
        elif choice == '3':     #Isolate and save the mask
            cv2.imshow('input',raw_frame)
            isolate_mask(raw_frame)
            closing_tasks()
        
        elif choice == '4':     #Calculte internal standard deviation 
            cv2.imshow('input',raw_frame)
            new_read = return_isolate_mask(raw_frame)
            print(new_read)
            cv2.imshow('current mask', new_read)
            
            for i in range(new_read.shape[0]):
                for j in range(new_read.shape[1]):
                    
                    if new_read[i,j,0] == new_read[i,j,1]:
                        if new_read[i,j,1] == new_read[i,j,2]:
                            pass
                    else :
                        b_list.append(new_read[i,j,0])
                        g_list.append(new_read[i,j,1])
                        r_list.append(new_read[i,j,2])

            b_sd = sd_calc(b_list)
            #g_sd = sd_calc(g_list)   
            #r_sd = sd_calc(r_list)  
            print('Standard Deviation : {0}'.format(b_sd) )  
            print('\n') 

            closing_tasks()


        elif choice == '5':     #Find the mean and deviation from the 3 samples 
            print('Enter the 3 photos')
            img1 = input ('Enter the first photo\n')
            img2 = input ('Enter the second photo\n')
            img3 = input ('Enter the third photo\n')
            img1_fish_name = 'fish' + str(img1) + '.png' 
            img2_fish_name = 'fish' + str(img2) + '.png' 
            img3_fish_name = 'fish' + str(img3) + '.png' 
            img1_fish = cv2.imread(img1_fish_name)
            img2_fish = cv2.imread(img2_fish_name)
            img3_fish = cv2.imread(img3_fish_name)
            sample_mean , sample_sd = choice5(img1_fish,img2_fish,img3_fish)
            closing_tasks()

        elif choice == '5.1':    #Show sampled choices
            print('The sampled mean is : '+ str(sample_mean ))
            print('The sampled deviation is : '+ str(sample_sd) )
            
        
        elif choice == '6':
            black_white = return_isolate_mask(raw_frame)
            output = superisolate(black_white)
            blur_constant = cv2.Laplacian(output, cv2.CV_64F).var()
            print('Blur Index : {}'.format(blur_constant))
            cv2.imshow('Super - Isolated' , output)
            closing_tasks()


        elif choice == '7':

            black_white = return_isolate_mask(raw_frame)
            output = superisolate(black_white)
            cv2.imshow('Super - Isolated' , output)
            cv2.imwrite(process_name + '_superisolate.png',output)
            closing_tasks()


        #Closing The Software
        cv2.destroyAllWindows()







if __name__ == '__main__':
    main()
