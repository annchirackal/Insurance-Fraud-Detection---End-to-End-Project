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
from flask import Flask, render_template, request, redirect, url_for
# Initialize the logger
logger = logger.App_Logger()

app = Flask(__name__)

# Define the log file path
preprocessing_logger_path = "Insurance-Fraud-Detection---End-to-End-Project/Logs/preprocessingLog.txt"
clustering_logger_path = "Insurance-Fraud-Detection---End-to-End-Project/Logs/pre_training_clusteringlog.txt"
training_logger_path="Insurance-Fraud-Detection---End-to-End-Project/Logs/trainingLog.txt"
prediction_logger_path ="Insurance-Fraud-Detection---End-to-End-Project/Logs/predictionLog.txt"

def train_model(data):
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
    if not os.path.exists(directory):
        os.makedirs(directory)
        # Open the file in append mode
    with open(training_logger_path, 'a+') as log_file:
            Model_trainer=trainer.trainModel(file_object=log_file, logger_object=logger)
            Model_trainer.trainingModel(X,y)
    return str("Sucessfull trained the model")

    
def predict_model(data):
    directory = os.path.dirname(prediction_logger_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Open the file in append mode
    with open(prediction_logger_path, 'a+') as log_file:
        #data=pd.read_csv(constants.pred_file_path)
        pred_processor=prediction_preprocessing.prediction_Preprocessor(file_object=log_file,
                                                        logger_object=logger)
        data=pred_processor.prediction_preprocssing_pipeline(data)
        #data.drop(columns=["fraud_reported"],inplace=True)
        model_loader=file_methods.File_Operation(file_object=log_file,
                                                        logger_object=logger)
        model=model_loader.load_model()
    
        
        results,pred_prob=pred_processor.predict_results(model=model,data=data)
        data["predicted_labels"]=results
        data["Prediction_probabilities"]=pred_prob
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        option = request.form.get('option')
        file = request.files['file']

        if file:
            data = pd.read_csv(file)
            
            if option == 'train':
                file_path = os.path.join(constants.input_file_path)
                data.to_csv(file_path,index=False)
                r_message =train_model(data)
                return render_template('success.html', message=r_message)
            elif option == 'test':
                prediction_input= data
                results = predict_model(data)
                results=results[["predicted_labels","Prediction_probabilities"]].join(prediction_input)
                results.to_csv(constants.predicted_results_path)
                return_results=results[["policy_number","predicted_labels","Prediction_probabilities"]]
                results_dict = return_results.to_dict(orient='records')
                columns = return_results.columns.tolist()
                return render_template('success.html', message="Prediction Results:", data_dict=results_dict, columns=columns)
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

    
    
 
    


    


