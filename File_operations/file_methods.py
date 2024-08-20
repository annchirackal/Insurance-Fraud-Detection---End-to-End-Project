import pickle
import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))
from constants import constants


class File_Operation:
    """
                This class shall be used to save the model after training
                and load the saved model for prediction.

                """
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory='Insurance-Fraud-Detection---End-to-End-Project/models/'
        self.filename=constants.filename
    
    

    def save_model(self,model):
        """
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the save_model method of the File_Operation class')
        
        try:
            path = os.path.join(self.model_directory,self.filename) #create seperate directory for each cluster
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path) #
            with open(path +'/' + self.filename+'.sav',
                      'wb') as f:
                pickle.dump(model, f) # save the model to file
            self.logger_object.log(self.file_object,
                                   'Model File '+self.filename+' saved. Exited the save_model method of the Model_Finder class')

            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model File '+self.filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception()

    def load_model(self):
        """
                    Method Name: load_model
                    Description: load the model file to memory
                    Output: The Model file loaded in memory
                    On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the load_model method of the File_Operation class')
        try:
            with open(self.model_directory + self.filename + '/' + self.filename + '.sav',
                      'rb') as f:
                self.logger_object.log(self.file_object,
                                       'Model File ' + self.filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + self.filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()
