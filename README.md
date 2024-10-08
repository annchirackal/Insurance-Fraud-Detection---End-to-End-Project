# Insurance-Fraud-Detection---End-to-End-Project

### Problem Statement

Develop an end-to-end web application for classifying insurance claims as fraudulent or legitimate so that users can easily approve the legitimate cases and intiate a deatiled ivestigation on cases which are calssified as fraud. The application should:

- Model Training Option:
  - Allow users to upload labeled data (CSV files) to retrain the classification model.
  - Save the uploaded data to a specified folder and use it to update the model’s training.
- Prediction:
  - Enable users to upload new, unlabeled CSV files for prediction.
  - Provide the ability to view the predicted results and whether each claim is classified as fraudulent or legitimate.
- User Interface:
  - Include a user-friendly interface with options to:
  - Choose between training the model or making predictions.
  - Upload corresponding CSV files based on the selected option.
  - Display success messages and results clearly on the web interface.
  - Ensure seamless navigation between different functionalities of the app.

- Model Management:
  - Provide feedback on the model’s performance after training and predictions.
  - Allow users to initiate model retraining with new data as needed.
### Design and Roadmap
![alt text](<Road Map.jpeg>)

### Data Description
The data contains the following attributes:
Features:
- months_as_customer: It denotes the number of months for which the customer is associated with the insurance company.
- age: continuous. It denotes the age of the person.
- policy_number: The policy number.
- policy_bind_date: Start date of the policy.
- policy_state: The state where the policy is registered.
- policy_csl-combined single limits. How much of the bodily injury will be covered from the total damage.
https://www.berkshireinsuranceservices.com/arecombinedsinglelimitsbetter  
- policy_deductable: The amount paid out of pocket by the policy-holder before an insurance provider will pay any expenses.
- policy_annual_premium: The yearly premium for the policy.
- umbrella_limit: An umbrella insurance policy is extra liability insurance coverage that goes beyond the limits of the insured's homeowners, auto or watercraft insurance. It provides an additional layer of security to those who are at risk of being sued for damages to other people's property or injuries caused to others in an accident.
- insured_zip: The zip code where the policy is registered.
- insured_sex: It denotes the person's gender.
- insured_education_level: The highest educational qualification of the policy-holder.
- insured_occupation: The occupation of the policy-holder.
- insured_hobbies: The hobbies of the policy-holder.
- insured_relationship: Dependents on the policy-holder.
- capital-gain: It denotes the monitory gains by the person.
- capital-loss: It denotes the monitory loss by the person.
- incident_date: The date when the incident happened.
- incident_type: The type of the incident.
- collision_type: The type of collision that took place.
- incident_severity: The severity of the incident.
- authorities_contacted: Which authority was contacted.
- incident_state: The state in which the incident took place.
- incident_city: The city in which the incident took place. 
- incident_location: The street in which the incident took place.
- incident_hour_of_the_day: The time of the day when the incident took place.
- property_damage: If any property damage was done.
- bodily_injuries: Number of bodily injuries.
- Witnesses: Number of witnesses present.
- police_report_available: Is the police report available.
- total_claim_amount: Total amount claimed by the customer.
- injury_claim: Amount claimed for injury
- property_claim: Amount claimed for property damage.
- vehicle_claim: Amount claimed for vehicle damage.
- auto_make: The manufacturer of the vehicle
- auto_model: The model of the vehicle. 
- auto_year: The year of manufacture of the vehicle. 

- fraud_reported:  Y or N(Target Label)
Whether the claim is fraudulent or not.
### Objectives:
- Implement a robust model for detecting insurance fraud.
- Improve recall to reduce the impact of false negatives.
- Provide an intuitive web interface for easy model training and prediction.
## Features Included:
- Model Training: Retrains using XGBoost and Random Forest, selecting the best-performing model.
- Prediction: Predicts fraudulent claims with 95% accuracy.
- Web Interface: Allows users to choose between training and prediction modes and upload CSV files for data input.
### Prerequisites/Steps For Execution:
- Python 3.10 or above
- Install dependencies: pip install -r requirements.txt
- Run the app: python main.py
### Technologies Used:
- Flask: Used for developing the web application interface.
- Python: Primary programming language for backend development.
- Scikit-learn: Implements KNN, Random Forest, and XGBoost models.

### Challenges Faced:
- Model Selection: Balancing between XGBoost and Random Forest for optimal performance.
- Recall Optimization: Enhancing recall to minimize false negatives.
- User Interface: Designing an easy-to-use web interface for model interaction.



