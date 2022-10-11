# -*- coding: utf-8 -*-
"""
PHYS20161 Introduction to Programming
Assessment 1: Bouncy Ball

Analyses the bounces of a ball over a minimum height. Requires inputs of
for initial height, minimum height, and the efficiency of the ball's bounce. 
The efficiency is the factor by which the ball loses energy, so that the height
of the next bounce is given by:
    
    next height = current height * efficiency.

Outputs the number of bounces over the minimum height, the time taken for 
the ball to reach the top of the last bounce above minimum height, and a graph 
of the heights and times for every peak above minimum height.


Timothé Rhein 
ID: 10139740
18/10/2019
"""
import numpy as np
import matplotlib.pyplot as plt

def bounce_time(height, efficiency) :
    """
    The time taken for the ball to go from one peak of height 'h' to the 
    next peak of height 'h*efficiency'. 
    """
    fall_time = np.sqrt(2*height/g)
    rise_time = np.sqrt(2*height*efficiency/g)
    
    return fall_time + rise_time
 
#constants
g = 9.80665 #standard gravitational acceleration defined by CGPM (1901)

while True : #runs the code until the correct inputs are given
    
    try :
        #inputs - must be floats
        initial_height = float(input('What is the initial height the ball is dropped from in meters? '))
        minimum_height = float(input('What is the minimum height in meters? '))
        efficiency = float(input("What is the efficiency of the ball's bounce? "))
        
        #Error checking
        if initial_height <= 0 or minimum_height <= 0 :
            print('\nMake sure the heights are greater than 0.')
            print('Please try again.\n') 
            
        elif minimum_height >= initial_height :
            print('\nPlease set the minimum height to be less than the initial height.')
            print('Please try again.\n')
        
        elif efficiency == 1 :
            print('\nThe ball keeps on bouncing indefinitely.')
            print('Please try again.\n')
        
        elif efficiency == 0 :
            print('\nThe ball does not bounce.')
            print('Please try again.\n')
                    
        elif efficiency > 1 or efficiency < 0 :
            print('\nThe efficiency must be greater than 0 and less than 1.')
            print('Please try again.\n')
    
        elif initial_height * efficiency < minimum_height :
            print('\nThere are no bounces above minimum height.')
            print('Please try again.\n')
            
        else :      #iterate for each jump above minimum_height
            
            #initial values for while function iteration
            bounces = 0
            total_time = 0
            height = initial_height
            array_heights = [initial_height]
            array_times = [total_time]
            
            while height * efficiency >= minimum_height :
                
                #number of bounces
                bounces = bounces + 1
            
                #time taken for the bounces
                total_time = total_time + bounce_time(height,efficiency)
                
                #move to next bounce
                height = height * efficiency
                
                #add values for to the arrays of all bounces
                array_heights.append(height)
                array_times.append(total_time)
                
            
            #Graph of the height of the peaks and the time at which the ball reaches them
            plt.plot(array_times , array_heights , 'ro')
            plt.xlabel('Time (s)')
            plt.ylabel('Height of peaks (m)')
            plt.title('Heights and times of each peak above the minimum height')
            plt.show()
            
            print('\nThe total number of bounces above the minimum height is ',bounces,'.')
            print('The total amount of time to complete these bounces is {0:0.4g}'.format(total_time),'seconds.')
            
            break # stops while function after correct outputs
        
    #Error testing - make sure all inputs are floats        
    except ValueError :
        
        print('\nMake sure all inputs are floats.')
        print('Please try again.\n')