"""
This is the Entry point for Training the Machine Learning Model.

"""

# Doing the necessary imports
from sklearn.model_selection import train_test_split
import sys
import os

# Add the path to the 'App/Application_logging' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))
from application_logging.logger import App_Logger
from File_operations.file_methods import File_Operation
from Model_Training.tuner import Model_Finder
from constants import constants
import numpy as np
import pandas as pd

class trainModel:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.log_writer = logger_object
    
    def trainingModel(self,X,Y):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
              
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3, random_state=355)

            model_finder=Model_Finder(self.file_object,self.log_writer) # object initialization
           

            #getting the best model for each of the clusters
            best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)
            #train_df_sample.to_csv(constants.path_to_save_training_data,index=False)
                #saving the best model to the directory.
            file_op = File_Operation(self.file_object,self.log_writer)
            save_model=file_op.save_model(best_model)

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training:'+ str(e))
            self.file_object.close()
            raise Exception