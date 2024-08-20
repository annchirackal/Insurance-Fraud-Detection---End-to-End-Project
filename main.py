from application_logging import logger
from Data_Preprocessing import preprocessing
from Clustering.clustring_data import KMeansClustering
from constants import constants
from Model_Training import trainer
from Prediction import prediction_preprocessing
import os
import warnings
import pandas as pd
from File_operations import file_methods
warnings.filterwarnings("ignore")
# Initialize the logger
logger = logger.App_Logger()



# Define the log file path
preprocessing_logger_path = "Insurance-Fraud-Detection---End-to-End-Project/Logs/preprocessingLog.txt"
clustering_logger_path = "Insurance-Fraud-Detection---End-to-End-Project/Logs/pre_training_clusteringlog.txt"
training_logger_path="Insurance-Fraud-Detection---End-to-End-Project/Logs/trainingLog.txt"
prediction_logger_path ="Insurance-Fraud-Detection---End-to-End-Project/Logs/predictionLog.txt"

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
    

# # Ensure the directory exists and open the file in append mode
# directory = os.path.dirname(clustering_logger_path)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# # Open the file in append mode
# with open(clustering_logger_path, 'a+') as log_file:
#     KM_Cluster=KMeansClustering(file_object=log_file, logger_object=logger)
#     X=KM_Cluster.run_clustering_data(data=X)
#     clustered_df=X
#     clustered_df["labels"]=y
#     ### Save the clusted data
#     # Define the path
#     path = "Insurance-Fraud-Detection---End-to-End-Project/Data/clustered_data.csv"
#     # Now save the clustered data to the file
#     clustered_df.to_csv(path, index=False)

# directory = os.path.dirname(training_logger_path)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# # Open the file in append mode
# with open(training_logger_path, 'a+') as log_file:
#     Model_trainer=trainer.trainModel(file_object=log_file, logger_object=logger)
    
#     Model_trainer.trainingModel(X,y)

directory = os.path.dirname(prediction_logger_path)
if not os.path.exists(directory):
    os.makedirs(directory)
# Open the file in append mode
with open(prediction_logger_path, 'a+') as log_file:
    data=pd.read_csv(constants.pred_file_path)
    pred_processor=prediction_preprocessing.prediction_Preprocessor(file_object=log_file,
                                                     logger_object=logger)
    data=pred_processor.prediction_preprocssing_pipeline(data.head())
    data.drop(columns=["fraud_reported"],inplace=True)
    model_loader=file_methods.File_Operation(file_object=log_file,
                                                     logger_object=logger)
    model=model_loader.load_model()
   
    
    results,pred_prob=pred_processor.predict_results(model=model,data=data)
    data["predicted_labels"]=results
    data["Prediction_probabilities"]=pred_prob
    data.to_csv("predicted_data.csv")

    
    
 
    


    


