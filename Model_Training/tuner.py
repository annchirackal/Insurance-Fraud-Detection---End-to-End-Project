from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score,recall_score
import pandas as pd

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
             
                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rf_classifier=RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)

    # def get_feature_importance(self,model,X_train,y_train, feature_names=None):
    #     """
    #     Get the feature importance from a trained model.

    #     Parameters:
    #     model : A trained machine learning model (e.g., RandomForest, LogisticRegression)
    #     feature_names : List of feature names (default is None). If provided, it will be used for the output.

    #     Returns:
    #     A pandas DataFrame with feature names and their corresponding importance scores.
    #     """
    #     self.logger_object.log(self.file_object, 'Entered the get_feature_impotance method under Model_Finder  class')
    #     try:
    #         model.fit(X_train, y_train)
    #         # Check if the model has feature_importances_ (tree-based models)
    #         if hasattr(model, 'feature_importances_'):
    #             importance = model.feature_importances_
    #         # Check if the model has coef_ (linear models)
    #         elif hasattr(model, 'coef_'):
    #             importance = np.abs(model.coef_.ravel())
    #         else:
    #             raise ValueError("The model does not have feature_importances_ or coef_ attributes.")

    #         if feature_names is None:
    #             feature_names = [f'Feature_{i}' for i in range(len(importance))]

    #         # Create a DataFrame for better readability
    #         self.feature_importance_df = pd.DataFrame({
    #             'Feature': feature_names,
    #             'Importance': importance
    #         }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    #         print(self.feature_importance_df )
    #         return self.feature_importance_df
    #     except Exception as e:
    #         self.logger_object.log(self.file_object,
    #                                'Exception occured in get_feature_impotance the Model_Finder class. Exception message:  ' + str(
    #                                    e))
    #         self.logger_object.log(self.file_object,
    #                                'Get feature importance failed .Exited get_feature_importance method of the Model_Finder class')
    #         raise Exception()
    # def get_features_to_drop(self,importance_df,threshold_score=50):
    #     """
    #     get a list of columns with low feature importance scores

    #     Parameters:
    #     df : A dataframe with feature names and feature scores
    #     Returns:
    #     A list with feature importance score less than given threshhold
    #     On Failure: Raise Exception
    #     """
    #     self.logger_object.log(self.file_object, 'Entered the get_features_to_drop method under Model_Finder  class')
    #     try:
    #         self.low_importance_features = importance_df[importance_df['Importance'] < threshold_score]['Feature'].tolist()
    #         self.logger_object.log(self.file_object,'Exited the get_features_to_drop method of the Model_Finder Features to drop are:'+str(self.low_importance_features) )
    #         return self.low_importance_features
    #     except Exception as e:
    #         self.logger_object.log(self.file_object,
    #                                'Exception occured in get_features_to_drop the Model_Finder class. Exception message:  ' + str(
    #                                    e))
    #         self.logger_object.log(self.file_object,
    #                                'get_features_to_drop failed .Exited get_features_to_drop method of the Model_Finder class')
    #         raise Exception()

    def get_best_params_for_rf(self,train_x,train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the SVM Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                    'n_estimators': [50,100 ],
                    'max_depth': [None, 5, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                           }

            #Creating an object of the Grid Search class
            self.grid_rf = GridSearchCV(estimator=self.rf_classifier, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid_rf.fit(train_x, train_y)

            #extracting the best parameters
            self.n_estimators = self.grid_rf.best_params_['n_estimators']
            self.max_depth = self.grid_rf.best_params_['max_depth']
            self.min_samples_split = self.grid_rf.best_params_['min_samples_split']
            self.min_samples_leaf= self.grid_rf.best_params_['min_samples_leaf']


            #creating a new model with the best parameters
            self.rf_classifier = RandomForestClassifier(n_estimators=self.n_estimators,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                   random_state=42)
            # training the mew model
          
            # self.feature_importance_df=self.get_feature_importance(self.rf_classifier,train_x,train_y,train_x.columns)
            # self.features_drop=self.get_features_to_drop(self.feature_importance_df)
            # self.rf_train_x=train_x.drop(columns=self.features_drop)
            # self.rf_classifier.fit(self.rf_train_x, train_y)
            self.rf_classifier.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'random Forest best params: '+str(self.grid_rf.best_params_)+'. Exited the get_best_params_for_rf method of the Model_Finder class')
            return self.rf_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_rf method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'rf training  failed. Exited the get_best_params_for_rf method of the Model_Finder class')
            raise Exception()
    import pandas as pd

   
    
    def get_best_params_for_xgboost(self,train_x,train_y):

        """
         Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
        Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                "n_estimators": [100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(8, 10, 1)

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']


            # creating a new model with the best parameters
            self.xgb = XGBClassifier(criterion=self.criterion, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1 )
            # training the mew model
            # self.feature_importance_df=self.get_feature_importance(self.rf_classifier,train_x,train_y, feature_names=train_x.columns)
            # self.features_drop=self.get_features_to_drop(self.feature_importance_df)
            # self.xgb_train_x=train_x.drop(columns=self.features_drop)
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) +'XGB Features_used:'+'. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.xgboost_recall_score=recall_score(test_y,self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score)+"Recall Score:"+str(self.xgboost_recall_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.xgboost_recall_score=recall_score(test_y,self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)+"Recall Score:"+str(self.xgboost_recall_score)) # Log AUC

            # create best model for Random Forest
            self.rf=self.get_best_params_for_rf(train_x,train_y)
            self.prediction_rf=self.rf.predict(test_x) # prediction using the 

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.rf_score = accuracy_score(test_y,self.prediction_rf)
                self.rf_recall_score=recall_score(test_y,self.prediction_rf)# recall score
                self.logger_object.log(self.file_object, 'Accuracy for rf:' + str(self.sv_score)+"Recall Score:"+str(self.rf_recall_score))
            else:
                self.rf_score = roc_auc_score(test_y, self.prediction_rf) # AUC for Random Forest
                self.rf_recall_score=recall_score(test_y,self.prediction_rf)# recall score
                self.logger_object.log(self.file_object, 'AUC for rf:' + str(self.rf_score)+"Recall Score:"+str(self.rf_recall_score))

            #comparing the two models
            if(self.rf_score <  self.xgboost_score):
                self.logger_object.log(self.file_object,'Sucessfully completed training retured XGB Model'
                                   )
                return 'XGBoost',self.xgboost
            else:
                self.logger_object.log(self.file_object,'Sucessfully completed training returedRandom forest model'
                                   )
                return 'Random Forest',self.rf_classifier

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

