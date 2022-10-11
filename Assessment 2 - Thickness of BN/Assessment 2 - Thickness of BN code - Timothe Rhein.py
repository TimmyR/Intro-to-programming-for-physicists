# -*- coding: utf-8 -*-
"""
PHYS20161 Introduction to Programming
Assessment 2: Thickness of Boron Nitride (BN)

This code reads a data file (Tunnelling_data_BN.csv) and outputs a graph of the
valid data points(since the data file contains errors) along with a fit by 
adjusting the one parameter.

The data gives the transmission coefficient with errors for different energies.
The relationship between transmission coefficient and energies is given by
Equation (3) of the Assignment 2: Thickness of Boron Nitride (BN) script, which
can be found at:
    https://online.manchester.ac.uk/bbcswebdav/pid-7510360-dt-content-rid-34778358_1/courses/I3133-PHYS-20161-1191-1SE-011049/BN%20script.pdf
    
The parameter being fitted to is the thickness of BN. The code outputs the 
thickness with an uncertainty, as well as the reduced chi-squared of the fit
and the amount of layers of BN. A graph of transmission coefficient against
energy shows both the fit and the data with error bars.

Timothé Rhein 
ID: 10139740
22/11/2019
"""
import numpy as np
import matplotlib.pyplot as plt

'''
####
CONSTANTS
####
'''

'''As defined in the script'''
V_0 = 3                             #in eV
epsilon_0 = 5.53e-3                 #(electic permitivity of free space) over e^2 in eV per Angstrom
epsilon_r = 4                       #relative electric permitivity of BN
sqrt_2m_over_hbar = 0.512317        # sqrt(2m) / h_bar in units of  eV^-1/2 A^-1
thickness_per_layer = 3             #3 Angstroms per layer
lambda_constant = np.log(2) / (8 * np.pi * epsilon_r * epsilon_0)
distance_1 = 1.2 * lambda_constant / V_0

'''Other constants'''
starting_thickness = 3      # approximate thickness of 1 layer of BN
step = 0.0001               # step added or subtracted to distance when reducing chi_squared
error_step = 0.1 * step     # step used to find the error in thickness


'''
####
FUNCTIONS
####
'''

'''File reading and validation functions'''

def float_check(value) :
    '''
    Checks if the value is a float. Returns True if it is, False if it isn't.
    '''
    try :
        float(value)
        return True
  
    except :
        return False


def read_file(filename) :
    '''
    Reads file and creates a vertically stacked array with all the data in 
    strings, omitting the column headers. Returns the data array.   
    '''
            
    input_file = open(filename,'r')
    
    skipped_column_headers = False
    data = np.empty((0,3))
    
    for line in input_file :
       
        if skipped_column_headers is False :            
            skipped_column_headers = True
            
        else :            
            split_up = line.split(',')
            split_up_array = np.array([split_up[0],split_up[1],split_up[2]])
            data = np.vstack((data,split_up_array))
            
    input_file.close()

    return data


def data_validation(data) :
    '''
    Validates each line and returns True if the condition for valid data are 
    met. The conditions are:
        - All values must be floats.
        - 0 <= transmission coefficient <= 1
        - 0 <= energy <= V_0
        - uncertainty > 0
        
    '''
    if float_check(data[i][0]) and float_check(data[i][1]) and float_check(data[i][2]) :
            if float(data[i][0]) >= 0 and float(data[i][0]) <= 1 :
                if float(data[i][1]) >= 0 and float(data[i][1]) <= V_0 :
                    if float(data[i][2]) > 0 :
            
                        return True


'''Fitted equations'''

def distance_2(thickness) :
    '''
    Returns the value of distance_2, given the equation:
        distance_2 = thickness - distance_1.
    '''
    distance_2 = thickness - distance_1
    
    return distance_2


def average_potential(thickness) :
    '''
    Returns the value of the average potential. Follows the equation:
        average_potential = V_0 - (1.15 * lambda_constant * ln(distance_2**2 / distance_1**2)) / (distance_2 - distance_1) .
    '''
    ln_term = np.log(distance_2(thickness)**2 / distance_1**2)
    first_term = 1.15 * lambda_constant / (distance_2(thickness) - distance_1)
    average_potential = V_0 - first_term * ln_term
    
    return average_potential


def transmission_coefficient(energy,thickness) :
    '''
    Returns the function describing the approximate relationship between the 
    transmission coefficient and energy. 
    '''
    sqrt_term = np.sqrt(average_potential(thickness) - energy)
    exponential_term = -2 * (distance_2(thickness) - distance_1) * sqrt_2m_over_hbar * sqrt_term
    transmission_coefficient = np.exp(exponential_term)  
    
    return transmission_coefficient


'''Chi-squared and error analysis'''

def chi_squared(data, thickness) :
    '''
    Returns the value of chi-squared for the the fit, depending on the 
    thickness. Follows the equation:
        chi_squared = sum( (difference/uncertainty)**2 )
    '''
    difference = data[:,0] - transmission_coefficient(data[:,1],thickness)      
    uncertainty = data[:,2]
    chi_squared = np.sum(np.square(difference / uncertainty))
    
    return chi_squared    
   
    
def reduced_chi_squared(data, thickness, number_of_parameters) :
    '''
    Returns the reduced chi^2 given the chi^2 and the number of parameters.
    '''
    number_of_degrees_of_freedom = len(data) - number_of_parameters
    reduced_chi_squared = chi_squared(data,thickness) / number_of_degrees_of_freedom
    
    return reduced_chi_squared


def function_to_be_minimised(data,thickness) :
    '''
    The thickness at which this function is minimised corresponds to the 
    thickness where chi^2 is equal to the minimum chi^2 + 1. The difference
    between this thickness and the final_thickness (for which chi^2 is minimised)
    corresonds to the error in thickness.
    '''
    function = np.square( chi_squared(data,thickness) - chi_squared(data,final_thickness) - 1)
    
    return function


def error_in_thickness(data,thickness,step) :
    '''
    Finds the thickness at which the function_to_be_minimised is minimised. The
    error in thickness is then the absolute value of the difference between 
    this thickness and the final_thickness.
    '''
    
    
    while True:
        
        if np.isnan(function_to_be_minimised(data,thickness)) :
            thickness += step
        
        elif function_to_be_minimised(data, thickness + step) < function_to_be_minimised(data, thickness) :
            thickness += step
            
        elif function_to_be_minimised(data, thickness - step) < function_to_be_minimised(data, thickness) :
            thickness -= step
        
        else :
            break
        
    error_in_thickness = abs(thickness - final_thickness)
    
    return error_in_thickness
    

'''Other functions'''

def number_of_layers(thickness,thickness_per_layer) :
    '''
    Outputs the number of layers, rounded to the closest integer.
    '''
    number_of_layers = round (thickness / thickness_per_layer)
    
    return number_of_layers
    
    

'''
####
MAIN CODE
####
'''   

'''Read file'''

#read file data
try :    
    data = read_file('Tunnelling_data_BN.csv')
    file_read = True

#error message if unable to read file    
except :
    print('\nFile could not be read. Please try again.')
    file_read = False
    
'''Validate data'''
    
if file_read is True :
    
    #Make an array with only the valid data points where the right conditions are met
    valid_data = np.empty((0,3))
    
    for i in range(len(data)) :
        
        if data_validation(data) is True :
            
            temporary_valid_data = np.array([float(data[i][0]),float(data[i][1]),float(data[i][2])])
            valid_data = np.vstack((valid_data,temporary_valid_data))
      
        
    '''Finding the thickness'''

    #Iterate thickness in order to minimise chi squared
    
    thickness = starting_thickness   
    
    while True: # runs loop until 'break' when chi_squared is minimised
                
        if np.isnan(chi_squared(valid_data,thickness)) :    #for low enough thicknesses, chi_squared returns 'nan'.
            thickness += step
            
        elif chi_squared(valid_data,thickness + step) < chi_squared(valid_data,thickness) :
            thickness += step
            
        elif chi_squared(valid_data,thickness - step) < chi_squared(valid_data,thickness) :
            thickness -= step
            
        else :
            break # stop once chi_squared is minimized
    
    final_thickness = thickness 
    
    
    '''Code outputs'''
    
    #Print statements
    print('\nThe thickness of BN in this sample is {:5.3f} \u00B1 {:5.3f} Å.'.format(final_thickness,error_in_thickness(valid_data,thickness,error_step))) # formatted to 4 s.f.
    print('The reduced chi-squared is {:4.2f}.'.format(reduced_chi_squared(valid_data, final_thickness, 1)))
    print('There is {} layer(s) of BN.'.format(number_of_layers(final_thickness, thickness_per_layer))) #prints number of layers as integer

    #Plot data
    plt.plot(valid_data[:,1], transmission_coefficient(valid_data[:,1],final_thickness), color = 'r',
             label = 'Fit - Reduced chi-squared = {:4.2f}'.format(reduced_chi_squared(valid_data, final_thickness, 1)))
    plt.errorbar(valid_data[:,1], valid_data[:,0], yerr = valid_data[:,2], fmt = 'o', label = 'Data')
    plt.title('Transmission coefficient against Energy', fontsize = 20, fontname = 'Arial')
    plt.xlabel('Energy (eV)', fontsize = 14, fontname = 'Arial')
    plt.ylabel('Transmission coefficient', fontsize = 14, fontname = 'Arial')
    plt.grid(dashes = [8,4])
    plt.legend()
    plt.show()
    
