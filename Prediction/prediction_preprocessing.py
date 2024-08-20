import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))
from constants import constants
from Data_Preprocessing import preprocessing

class prediction_Preprocessor:
    """
    This class shall be used to clean and transform the data before training.

    Written By: annchirackal
    Version: 1.0
    Revisions: None
    """
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
   
        self.logger_object.log(self.file_object,
                                'Entered the preprocessor training class ')
    def ensure_matching_columns(self,train_df, pred_df):
            """
            Ensure that all columns in the training DataFrame are present in the prediction DataFrame.
            
            Parameters:
            train_df : pandas DataFrame : The training dataset.
            pred_df : pandas DataFrame : The prediction dataset.
            fill_value : The value to fill in for missing columns (default is 0).
            
            Returns:
            pandas DataFrame : The prediction DataFrame with missing columns added.

            """
            self.logger_object.log(self.file_object,
                                    'Entered the ensure_matching_columns method of  preprocessor training class ')
            try:
                # Get the columns in the training set
                train_columns = train_df.columns
                
                # Get the columns in the prediction set
                pred_columns = pred_df.columns
                
                # Identify missing columns in the prediction set
                missing_columns = set(train_columns) - set(pred_columns)
                if len(missing_columns)>0:
                    raise ValueError("Missing column in data"+str( missing_columns))
                
                # Ensure the columns in the prediction set are in the same order as the training set
                
                self.logger_object.log(self.file_object,
                                    'Sucessfully exited ensure_matching_columns method of  preprocessor training class ')
                return missing_columns
            except Exception as e:
                self.logger_object.log(self.file_object,
                                    'Exception in ensure_matching_columns '+str(e))
                raise Exception()
   

    def add_features(self,model, input_df):
        """
        Ensure that all features used in the model training are present in the input DataFrame.
        Any missing features will be added to the DataFrame and filled with 0.
        
        Parameters:
        model : Trained XGBoost or RandomForest model.
        input_df : pandas DataFrame : The input data for prediction.

        Returns:
        pandas DataFrame : The input DataFrame with missing columns added and filled with 0.
        """
        self.logger_object.log(self.file_object,
                                    'Entered the add_features method of  preprocessor training class ')
        try:
            # Attempt to retrieve feature names based on the model type
            if hasattr(model, 'get_booster'):  # XGBoost
                feature_names = model.get_booster().feature_names
            elif hasattr(model, 'feature_importances_'):  # RandomForest (and most sklearn models)
                feature_names = model.feature_names_in_
            else:
                raise ValueError("The model type is not supported or does not have feature names.")

            # Identify missing columns in the input DataFrame
            missing_columns = set(feature_names) - set(input_df.columns)
            
            # Add missing columns with default value 0
            for col in missing_columns:
                input_df[col] = 0
            
            # Ensure the input DataFrame columns are in the same order as the model's feature names
            input_df = input_df[feature_names]
        
            return input_df
        except Exception as e:
            self.logger_object.log(self.file_object,
                                    'exception add_features'+str(e))
            raise Exception()
             

    
    def prediction_preprocssing_pipeline(self,data):
            """
            A function to process data before passing to model for prediction
            Parameters:
            data: A dataframe for prediction
            """
            try:
                training_data=pd.read_csv(constants.input_file_path)
                training_data=training_data.drop(columns='fraud_reported')
                missing_col=self.ensure_matching_columns(train_df=training_data, 
                                                        pred_df=data)
                if len(missing_col)>0:
                    raise ValueError("Missing column in data"+str( missing_columns))
                else:
                    data_preprocessor=preprocessing.Preprocessor(file_object=self.file_object, 
                                                                 logger_object=self.logger_object)
                    data=data_preprocessor.remove_unwanted_spaces_and_characters(data)
                    data=data_preprocessor.remove_columns(data,constants.columns_to_drop)
                    is_null_value_present,null_value_columns= data_preprocessor.is_null_present(data)
                    if is_null_value_present:
                        data= data_preprocessor.impute_missing_values(data,null_value_columns)
                    data=data_preprocessor.scale_numerical_columns(data)
                    self.processed_data=data_preprocessor.encode_categorical_columns(data)
                    return self.processed_data
            except Exception as e:
                self.logger_object.log(self.file_object,
                                    'Exception in prediction data preprocessing '+str(e))
                raise Exception()
    def predict_results(self,data,model):
        """
        A method to predict the labels- given data respresent fraud case or not
        parameters:
        data: data frame with imput for model prediction
        model : a trained model for prediction
        """
        self.logger_object.log(self.file_object,
                                    'Entered the predict_results method of  preprocessor training class ')
        try :
             data=self.add_features(model,data)
             
             predicted_lables=model.predict(data)
             predicted_probabilities = model.predict_proba(data)
             label_probabilities=predicted_probabilities[np.arange(len(predicted_lables)), predicted_lables]
             self.logger_object.log(self.file_object,
                                    ' Sucessfully predicted results ')
             return predicted_lables,label_probabilities
        except Exception as e:
                self.logger_object.log(self.file_object,
                                    'Exception in predict_results method'+str(e))
                raise Exception()

        

            
                    
            
