"""
PHYS20161 Introduction to Programming
Assessment 3: Nuclear decay

This code measures the decay constants and half-lives of Rubidium-79 (Rb-79) 
and Strontium-79 (Sr-79) by analysing data from an experiment that measures the
activity of Rb-79 over time. The data is compiled from two data files,
'Nuclear_data_1.csv' and 'Nuclear_data_2.csv', and removes any data points that
include NaNs or where the uncertainty is zero or where the percentage 
uncertainty is greater than 100%.

The decay constants are found by finding the values for which chi^2 is
minimised. Since there may be faulty measurements, after a preliminary fit is
performed any point that are more than 5 standard deviations is removed from
the data and the fit is performed once again without the outliers.

The equation for activity of Rb-79 is given by equation (4) in the script, 
which can be found at:
    https://online.manchester.ac.uk/bbcswebdav/pid-7510361-dt-content-rid-35005767_1/courses/I3133-PHYS-20161-1191-1SE-011049/nuclear%20script.pdf

The code outputs graphs of the preliminary fit, the final fit without outliers,
and a contour plot of chi^2 values depending on the decay constant. The code 
also prints the decay constants and half-lives of Rb-79 and Sr-79 with their 
uncertainties, as well as the reduced chi^2 of the final fit.

Timothé Rhein 
ID: 10139740
10/12/2019
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.optimize as opt
import sys

'''
####
CONSTANTS
####
'''

initial_number_sr_nuclei = 10**-6 * sc.Avogadro / 10**12# initial number of Sr nuclei in units of 10^12 nuclei

#Initial values based on previous research
INITIAL_DECAY_CONSTANT_RB = 0.0005  # units of (s^-1)
INITIAL_DECAY_CONSTANT_SR = 0.005  #units of (s^-1)




'''
####
FUNCTIONS
####
'''
 

###~~~ Reading and validation functions ~~~###

def read_and_validate(filename) :
    '''
    Reads the file defined by filename, converts all values into floats, and 
    removes rows where the error is zero or where the percentage error is 
    greater than 100%.
    
    - filename (string)
    '''
    data = np.genfromtxt(filename , comments = '%' , delimiter = ',')
    
    data_without_nan_rows = data[np.isfinite(data).all(axis=1)]
    data_without_zero_error = data_without_nan_rows[np.where(data_without_nan_rows[:,2] != 0)]
    data_without_large_error = data_without_zero_error[np.where(data_without_zero_error[:,2] / data_without_zero_error[:,1] < 1)]
    
    return data_without_large_error


def convert_time_to_seconds(data) :
    '''
    Converts the time column of the data from hours to seconds.
    
    - data (array of floats of size (n,3))
    '''
    data_in_seconds = data
    data_in_seconds[:,0] = data[:,0] * 3600
    
    return data_in_seconds


def combine_files(filename_1, filename_2) :
    '''
    Combines the data from both files into 1 array, and sorts the data in
    ascending order of time. The time values are converted into seconds.
    
    - filename_1 (string)
    - filename_2 (string)
    '''
    data_1 = read_and_validate(filename_1)
    data_2 = read_and_validate(filename_2)
    
    data_combined = np.concatenate((data_1, data_2))
    
    data_sort_order = np.argsort(data_combined[:,0])
    data_sorted = data_combined[data_sort_order]
    
    data_in_seconds = convert_time_to_seconds(data_sorted)
    
    return data_in_seconds


###~~~ Fitted equation ~~~###

def activity(time, decay_constants) :
    '''
    Equation describing the activity of a sample as a function of time. The two
    free parameters are the decay constants of Rb and Sr. Outputs the activity
    in TBq.
    
    - time (float)
    - decay_constants (array of floats of size (2,))
    '''
    decay_constant_rb = decay_constants[0]
    decay_constant_sr = decay_constants[1]
    
    first_term = initial_number_sr_nuclei * decay_constant_rb * decay_constant_sr / (decay_constant_rb - decay_constant_sr)
    second_term = np.exp(- decay_constant_sr * time) - np.exp(- decay_constant_rb * time)
    
    activity = first_term * second_term
    
    return activity


###~~~ Chi-squared and error analysis ~~~###

def chi_squared(decay_constants, data) :
    '''
    Returns the value of chi-squared for the the fit, depending on the 
    two decay constants to be fitted to. Follows the equation:
        chi_squared = sum( (difference/uncertainty)**2 )
        
    - decay_constants (array of floats of size (2,))
    - data (array of floats of size (n,3))
    '''    
    difference = data[:,1] - activity(data[:,0], decay_constants)      
    uncertainty = data[:,2]
    
    chi_squared = np.sum(np.square(difference / uncertainty))
    
    return chi_squared


def reduced_chi_squared(decay_constants, data, number_of_parameters) :
    '''
    Returns the reduced chi^2 given the chi^2 and the number of parameters.
    
    - decay_constants (array of floats of size (2,))
    - data (array of floats of size (n,3))
    - number_of_parameters (float)
    '''
    number_of_degrees_of_freedom = len(data) - number_of_parameters
    reduced_chi_squared = chi_squared(decay_constants, data) / number_of_degrees_of_freedom
    
    return reduced_chi_squared


def remove_outliers(decay_constants, data) :
    '''
    Removes outliers more than 5 standard deviation away from the preliminary 
    fit. Outputs an array of the data without the outliers.
    
    - decay_constants (array of floats of size (2,))
    - data (array of floats of size (n,3))
    '''
    offset = np.abs(data[:,1] - activity(data[:,0], decay_constants))
    data_without_outliers = data[np.where(offset < 5 * data[:,2])]
    
    return data_without_outliers


def decay_constants_mesh(decay_constants, data, resolution):
    '''
    Creates a mesh of the decay constants to be used for contour plotting. 
    Outputs an array of size (2,m,m) where m is the resolution, as in how many
    points will be plotted per axis.
    
    - decay_constants (array of floats of size (2,))
    - data (array of floats of size (n,3))
    - resolution (float)
    '''
    decay_constant_rb_range = np.linspace(decay_constants[0] * 19/20, decay_constants[0] * 21/20, resolution)    
    decay_constant_sr_range = np.linspace(decay_constants[1] * 19/20, decay_constants[1] * 21/20, resolution)      
    decay_constant_rb_mesh, decay_constant_sr_mesh = np.meshgrid(decay_constant_rb_range, decay_constant_sr_range)
    decay_constants_mesh = np.array([decay_constant_rb_mesh,decay_constant_sr_mesh])
    
    return decay_constants_mesh
    

def chi_squared_mesh(decay_constants_mesh, data) :
    '''
    Creates an array of chi^2 values to be plotted in a contour plot against
    a range of decay constants, given by the decay_constants_mesh. Outputs an
    array of size (m,m), where m is the resolution defined in 
    decay_constants_mesh(...).
    
    - decay_constants_mesh (array of floats of size (2,m,m))
    - data (array of floats of size (n,3))
    '''
    decay_constant_rb = decay_constants_mesh[0]
    decay_constant_sr = decay_constants_mesh[1]
    
    chi_squared_values = np.empty((0, len(decay_constant_rb)))

    for i in range(len(decay_constant_sr)) :
        
        chi_squared_row = np.array([])
        for j in range(len(decay_constant_rb)) :
            
            chi_squared_temp = chi_squared(decay_constants_mesh[:,i,j], data)
            chi_squared_row = np.append(chi_squared_row,chi_squared_temp)
            
        chi_squared_values = np.vstack((chi_squared_values,chi_squared_row))

    return chi_squared_values


def errors_on_parameters(contour) :
    '''
    Finds the error on the parameters by considering the contour where :
        chi^2 = chi^2_min + 1 .
    The error in the decay constant is approximately half the difference 
    between the minimum and maximum values of that decay constant along the 
    contour.
    
    - contour (array of floats of size (m,2))
    '''
    error_decay_constant_rb = (np.max(contour[:,0]) - np.min(contour[:,0])) / 2
    error_decay_constant_sr = (np.max(contour[:,1]) - np.min(contour[:,1])) / 2
    
    error_decay_constants = np.array([error_decay_constant_rb, error_decay_constant_sr])
    
    return error_decay_constants


###~~~ Other functions ~~~###
    
def half_lives(decay_constants) :
    '''
    Calculates and the half-lives of Rb and Sr in minutes from the 
    decay constants s^-1. Outputs the half-lives as an array.
    
    - decay_constants (array of floats of size (2,))    
    '''
    half_lives_seconds = np.log(2) / decay_constants
    half_lives_minutes = half_lives_seconds / 60
    
    return half_lives_minutes


def error_half_lives(decay_constants, error_decay_constants) :
    '''
    Finds the error on the half-lives in minutes, given that the percentage
    uncertainty of the half-life and the decay constant is the same.
    
    - decay_constants (array of floats of size (2,)) 
    '''
    error_half_lives = half_lives(decay_constants) * error_decay_constants / decay_constants
    
    return error_half_lives


'''
####
MAIN CODE
####
'''


###~~~ Read and validate data ~~~###

try :
    data = combine_files('Nuclear_data_1.csv', 'Nuclear_data_2.csv')  
    
except :
    sys.exit('The data files could not be read. Please try again.')
 

###~~~ Preliminary fit ~~~###

INITIAL_GUESS_DECAY_CONSTANTS = np.array([INITIAL_DECAY_CONSTANT_RB, INITIAL_DECAY_CONSTANT_SR])

try :
    decay_constants_preliminary = opt.fmin(chi_squared, INITIAL_GUESS_DECAY_CONSTANTS, args = (data,), disp = 0)
    print('\nThe preliminary fit was successful.')

except : 
    sys.exit('The fitting procedure has failed.')


###~~~ Removing outliers based onto preliminary fit ~~~###

data_without_outliers = remove_outliers(decay_constants_preliminary, data)


###~~~ Second fit without outliers ~~~###

#The starting values for this second fit are the decay constants found from the preliminary fit.
try :
    decay_constants = opt.fmin(chi_squared, decay_constants_preliminary, args = (data_without_outliers,), disp = 0)
    print('The final fit was successful.')
    
except :
    sys.exit('The fitting procedure has failed.')

    
###~~~ Plots of the data ~~~###

#Plot of preliminary data
figure = plt.figure()
axes = figure.add_subplot(111)

axes.errorbar(data[:,0], data[:,1], yerr = data[:,2], fmt = 'r.', label = 'Data')   #data points
axes.plot(data[:,0], activity(data[:,0], decay_constants_preliminary), 
          label = 'Preliminary fit \nReduced chi^2 = {:.2f}'.format(reduced_chi_squared(decay_constants_preliminary, data, 2)))   #fit line
axes.legend()
axes.grid(dashes = [8,4])
axes.set_xlabel('Time (seconds)', fontsize = 14, fontname = 'Arial')
axes.set_ylabel('Activity (TBq)', fontsize = 14, fontname = 'Arial')
axes.set_title('Activity of Rb-79 - preliminary fit', fontsize = 20, fontname = 'Arial')

plt.savefig('Preliminary fit of data.png', dpi = 300)
plt.show()

#Plot of data without outliers
figure = plt.figure()
axes = figure.add_subplot(111)

axes.errorbar(data_without_outliers[:,0], data_without_outliers[:,1], 
              yerr = data_without_outliers[:,2], fmt = 'g.', label = 'Data without outliers')   #data points
axes.plot(data_without_outliers[:,0], activity(data_without_outliers[:,0], decay_constants), 
          label = 'Final fit\nReduced chi^2 = {:.2f}'.format(reduced_chi_squared(decay_constants, data_without_outliers, 2)))   #fit line
axes.legend()
axes.grid(dashes = [8,4])
axes.set_xlabel('Time (seconds)', fontsize = 14, fontname = 'Arial')
axes.set_ylabel('Activity (TBq)', fontsize = 14, fontname = 'Arial')
axes.set_title('Activity of Rb-79 - final fit', fontsize = 20, fontname = 'Arial')

plt.savefig('Final fit of data.png', dpi = 300)
plt.show()


###~~~ Contour plot of chi^2 values ~~~###

#Mesh of decay constants and chi^2 values for contour plot
decay_constants_mesh = decay_constants_mesh(decay_constants, data_without_outliers, 250)
chi_squared_mesh = chi_squared_mesh(decay_constants_mesh, data_without_outliers)

figure = plt.figure()
axes = figure.add_subplot(111)

#Colour filled contour plot
axes.contourf(decay_constants_mesh[0], decay_constants_mesh[1], chi_squared_mesh,
              cmap = 'viridis')   

#Contour line of minimum chi^2 + 1 level shown as yellow dashed line on the plot                                                      
contour = axes.contour(decay_constants_mesh[0], decay_constants_mesh[1], chi_squared_mesh, 
                       levels = [chi_squared(decay_constants, data_without_outliers)+1], 
                       colors = 'yellow', linestyles = 'dashed')

axes.scatter(decay_constants[0], decay_constants[1], label = 'Minimum chi^2')   # location of minimised chi^2 value
contour.collections[0].set_label('Minimum chi^2 + 1')             # label of minimum chi^2 + 1 level
axes.legend()
axes.set_xlabel('Decay constant Rb (s^-1)', fontsize = 14, fontname = 'Arial')
axes.set_ylabel('Decay constant Sr (s^-1)', fontsize = 14, fontname = 'Arial')
axes.set_title('Chi^2 contour ', fontsize = 20, fontname = 'Arial')

plt.savefig('Contour plot of chi^2.png', dpi = 300)
plt.show()


###~~~ Errors on the parameters ~~~###

#Create array of points on the contour of minimum chi_squared + 1
chi_squared_plus_1 = contour.allsegs[0][0]

error_decay_constants = errors_on_parameters(chi_squared_plus_1)


###~~~ Print statements ~~~###
    
print('\nThe measured decay constant of Rubidium-79 is {:.6f} +/- {:.6f} s^-1.'.format(decay_constants[0], error_decay_constants[0]))
print('The measured decay constant of Strontium-79 is {:.5f} +/- {:.5f} s^-1.'.format(decay_constants[1], error_decay_constants[1]))

print('\nThe half-life of Rubidium-79 is {:.3} +/- {:.1} minutes.'.format(half_lives(decay_constants)[0], 
      error_half_lives(decay_constants, error_decay_constants)[0]))
print('The half-life of Strontium-79 is {:.3} +/- {:.1} minutes.'.format(half_lives(decay_constants)[1], 
      error_half_lives(decay_constants, error_decay_constants)[1]))

print('\nThe reduced chi^2 of the fit is {:.2f}.'.format(reduced_chi_squared(decay_constants, data_without_outliers, 2)))
