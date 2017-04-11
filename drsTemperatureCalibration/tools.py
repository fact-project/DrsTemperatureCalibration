import numpy as np
import os
import sys


def get_linear_fit_values(x_values, y_values, y_values_errors=[]):
    y_weighting = 1/pow(y_values_errors, 2)

    S_1 = np.sum(y_weighting)
    S_x = np.sum(y_weighting*x_values)
    S_xx = np.sum(y_weighting*pow(x_values, 2))

    S_y = np.sum(y_weighting*y_values)
    S_xy = np.sum(y_weighting*x_values*y_values)

    D = S_1*S_xx - pow(S_x, 2)

    var = [(-S_x*S_y + S_1*S_xy)*(1/D), (S_xx*S_y - S_x*S_xy)*(1/D)]
    cov = [[S_1/D, -S_x/D], [-S_x/D, S_xx/D]]

    return(var, cov)


####################################################################################################
####################################################################################################
# TODO check safety stuff. maybe remove
def safety_stuff(store_filename):
    if(os.path.isfile(store_filename)):
        userInput = input("File '"+str(store_filename)+"' allready exist.\n" +
                          " Type 'y' to overwrite File\nYour input: ")
        if(userInput != 'y'):
            sys.exit()

    elif(not os.path.isdir(store_filename[0:store_filename.rfind("/")])):
        print("Folder '", store_filename[0:store_filename.rfind("/")], "' does not exist")
        sys.exit()
