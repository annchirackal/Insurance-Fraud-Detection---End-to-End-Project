from application_logging import logger
from Data_Preprocessing import preprocessing
from constants import constants
import os

# Initialize the logger
logger = logger.App_Logger()

# Define the log file path
preprocessing_logger_path = "Logs/preprocessingLog.txt"

# Ensure the directory exists and open the file in append mode
directory = os.path.dirname(preprocessing_logger_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Open the file in append mode
with open(preprocessing_logger_path, 'a+') as log_file:
    # Initialize the Preprocessor with the file object and logger
    data_preprocess = preprocessing.Preprocessor(file_object=log_file, logger_object=logger)
    ## Run preprocess data method in Preprocessor class
    X,y=data_preprocess.preprocess_data(constants.input_file_path)
    

  
   
  
  
    


