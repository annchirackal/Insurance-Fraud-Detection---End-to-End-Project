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
    
    # Read input file
   
    df = data_preprocess.read_data(constants.input_file_path)
    df=data_preprocess.remove_unwanted_spaces_and_characters(data=df)
    df=data_preprocess.remove_columns(data=df,columns=constants.columns_to_drop)
    is_null_value_present,null_value_columns=  data_preprocess.is_null_present(data=df)
    if is_null_value_present:
        df= data_preprocess.impute_missing_values(data=df, cols_with_missing_values=null_value_columns)
    df=data_preprocess.scale_numerical_columns(data=df)
    df=data_preprocess.encode_categorical_columns(data=df)
    X,y=data_preprocess.separate_label_feature(data=df, label_column_name=constants.label_column_name)
    X,y=data_preprocess.handle_imbalanced_dataset(X,y)
    

    
  
    


