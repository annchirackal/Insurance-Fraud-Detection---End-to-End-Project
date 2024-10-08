import os
input_file_path="Insurance-Fraud-Detection---End-to-End-Project\Data\insuranceFraud.csv"
predicted_results_path="Insurance-Fraud-Detection---End-to-End-Project\Data\predicted_results.csv"

columns_list=['months_as_customer','age','policy_number', 
               'policy_bind_date','policy_state', 'policy_csl',
               'policy_deductable','policy_annual_premium', 'umbrella_limit', 
               'insured_zip','insured_sex','insured_education_level',
                'insured_occupation', 'insured_hobbies','insured_relationship',
                'capital-gains','capital-loss','incident_date', 
                'incident_type','collision_type','incident_severity', 
                'authorities_contacted','incident_state', 'incident_city', 
                'incident_location','incident_hour_of_the_day', 'number_of_vehicles_involved',
                'property_damage', 'bodily_injuries', 'witnesses',
                'police_report_available', 'total_claim_amount', 'injury_claim',
                'property_claim', 'vehicle_claim', 'auto_make', 
                'auto_model','auto_year', 'fraud_reported']

columns_to_drop =['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                  'incident_date','incident_state','incident_city','insured_hobbies','auto_make',
                  'auto_model','auto_year','age','total_claim_amount']
label_column_name='fraud_reported'
path_to_save_training_data=path = "Insurance-Fraud-Detection---End-to-End-Project/Data/sample_data_used_for_train.csv"
filename="Insutance_Fraud_Detection_Model"