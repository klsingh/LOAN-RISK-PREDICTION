import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('lending_club_data.csv')

# Drop columns with more than 50% missing values
data = data.dropna(thresh=int(0.5*len(data)), axis=1)

# Drop irrelevant columns
data = data.drop(['id', 'member_id', 'url', 'desc'], axis=1)

# Drop columns related to future or post loan
data = data.drop(['total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d'], axis=1)

# Encode categorical variables
data = pd.get_dummies(data, columns=['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
                      'verification_status', 'pymnt_plan', 'purpose', 'initial_list_status', 'application_type'])

# Fill missing values with median
data = data.fillna(data.median())
# Define the target variable
y = data['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0)

# Define the feature matrix
X = data.drop(['loan_status'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict on testing data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy score:', accuracy)

# Predict on new data
new_data = pd.DataFrame({'term_36 months': [0], 'term_60 months': [1], 'loan_amnt': [10000], 'funded_amnt': [10000], 'funded_amnt_inv': [10000], 'int_rate': [10.99], 'installment': [329.67], 'grade_A': [1], 'grade_B': [0], 'grade_C': [0], 'grade_D': [0], 'grade_E': [0], 'grade_F': [0], 'grade_G': [0], 'sub_grade_A1': [1], 'sub_grade_A2': [0], 'sub_grade_A3': [0], 'sub_grade_A4': [0], 'sub_grade_A5': [0], 'sub_grade_B1': [0], 'sub_grade_B2': [0], 'sub_grade_B3': [0], 'sub_grade_B4': [0], 'sub_grade_B5': [0], 'sub_grade_C1': [0], 'sub_grade_C2': [0], 'sub_grade_C3': [0], 'sub_grade_C4': [0], 'sub_grade_C5': [0], 'sub_grade_D1': [0], 'sub_grade_D2': [0], 'sub_grade_D3': [0], 'sub_grade_D4': [0], 'sub_grade_D5': [0], 'sub_grade_E1': [0], 'sub_grade_E2': [0], 'sub_grade_E3': [0], 'sub_grade_E4': [0], 'sub_grade_E5': [0], 'sub_grade_F1': [0], 'sub_grade_F2': [0], 'sub_grade_F3': [0], 'sub_grade_F4': [0], 'sub_grade_F5': [0], 'sub_grade_G1': [0], 'sub_grade_G2': [0], 'sub_grade_G3': [0], 'sub_grade_G4': [0], 'sub_grade_G5': [0], 'emp_length_1 year': [0], 'emp_length_10+ years': [1], 'emp_length_2 years': [0], 'emp_length_3 years': [0], 'emp_length_4 years': [0], 'emp_length_5 years': [0], 'emp_length_6 years': [0], 'emp_length_7 years': [0], 'emp_length_8 years': [0], 'emp_length_9 years': [0], 'emp_length_< 1 year': [0], 'home_ownership': ['OWN'], 'annual_inc': [60000], 'verification_status_Not Verified': [1], 'verification_status_Source Verified': [0], 'verification_status_Verified': [
                        0], 'purpose_car': [0], 'purpose_credit_card': [0], 'purpose_debt_consolidation': [1], 'purpose_home_improvement': [0], 'purpose_house': [0], 'purpose_major_purchase': [0], 'purpose_medical': [0], 'purpose_moving': [0], 'purpose_other': [0], 'purpose_renewable_energy': [0], 'purpose_small_business': [0], 'purpose_vacation': [0], 'purpose_wedding': [0], 'addr_state_AK': [0], 'addr_state_AL': [0], 'addr_state_AR': [0], 'addr_state_AZ': [0], 'addr_state_CA': [0], 'addr_state_CO': [0], 'addr_state_CT': [0], 'addr_state_DC': [0], 'addr_state_DE': [0], 'addr_state_FL': [0], 'addr_state_GA': [0], 'addr_state_HI': [0], 'addr_state_IA': [0], 'addr_state_ID': [0], 'addr_state_IL': [0], 'addr_state_IN': [0], 'addr_state_KS': [0], 'addr_state_KY': [0], 'addr_state_LA': [0], 'addr_state_MA': [0], 'addr_state_MD': [0], 'addr_state_ME': [0], 'addr_state_MI': [0], 'addr_state_MN': [0], 'addr_state_MO': [0], 'addr_state_MS': [0], 'addr_state_MT': [0], 'addr_state_NC': [0], 'addr_state_ND': [0], 'addr_state_NE': [0], 'addr_state_NH': [0], 'addr_state_NJ': [0], 'addr_state_NM': [0], 'addr_state_NV': [0], 'addr_state_NY': [0], 'addr_state_OH': [0], 'addr_state_OK': [0], 'addr_state_OR': [0], 'addr_state_PA': [0], 'addr_state_RI': [0], 'addr_state_SC': [0], 'addr_state_SD': [0], 'addr_state_TN': [0], 'addr_state_TX': [0], 'addr_state_UT': [0], 'addr_state_VA': [0], 'addr_state_VT': [0], 'addr_state_WA': [0], 'addr_state_WI': [0], 'addr_state_WV': [0], 'addr_state_WY': [0]})

# Predict on new data
new_data = pd.DataFrame({'term_36 months': [0], 'term_60 months': [1], 'loan_amnt': [10000], 'funded_amnt': [10000], 'funded_amnt_inv': [10000], 'int_rate': [10.99], 'installment': [329.67], 'grade_A': [1], 'grade_B': [0], 'grade_C': [0], 'grade_D': [0], 'grade_E': [0], 'grade_F': [0], 'grade_G': [0], 'sub_grade_A1': [1], 'sub_grade_A2': [0], 'sub_grade_A3': [0], 'sub_grade_A4': [0], 'sub_grade_A5': [0], 'sub_grade_B1': [0], 'sub_grade_B2': [0], 'sub_grade_F3': [0], 'sub_grade_F4': [0], 'sub_grade_F5': [0], 'sub_grade_G1': [0], 'sub_grade_G2': [0], 'sub_grade_G3': [0], 'sub_grade_G4': [0], 'sub_grade_G5': [0], 'home_ownership_ANY': [0], 'home_ownership_MORTGAGE': [1], 'home_ownership_OWN': [0], 'home_ownership_RENT': [0], 'annual_inc': [60000], 'verification_status_Not Verified': [1], 'verification_status_Source Verified': [0], 'verification_status_Verified': [0], 'purpose_car': [0], 'purpose_credit_card': [0], 'purpose_debt_consolidation': [1], 'purpose_home_improvement': [0], 'purpose_house': [0], 'purpose_major_purchase': [0], 'purpose_medical': [0], 'purpose_moving': [0], 'purpose_other': [0], 'purpose_renewable_energy': [0], 'purpose_small_business': [
                        0], 'purpose_vacation': [0], 'purpose_wedding': [0], 'addr_state_AK': [0], 'addr_state_AL': [0], 'addr_state_AR': [0], 'addr_state_AZ': [0], 'addr_state_CA': [0], 'addr_state_CO': [0], 'addr_state_CT': [0], 'addr_state_DC': [0], 'addr_state_DE': [0], 'addr_state_FL': [0], 'addr_state_GA': [0], 'addr_state_HI': [0], 'addr_state_IA': [0], 'addr_state_ID': [0], 'addr_state_IL': [0], 'addr_state_IN': [0], 'addr_state_KS': [0], 'addr_state_KY': [0], 'addr_state_LA': [0], 'addr_state_MA': [0], 'addr_state_MD': [0], 'addr_state_ME': [0], 'addr_state_MI': [0], 'addr_state_MN': [0], 'addr_state_MO': [0], 'addr_state_MS': [0], 'addr_state_MT': [0], 'addr_state_NC': [0], 'addr_state_ND': [0], 'addr_state_NE': [0], 'addr_state_NH': [0], 'addr_state_NJ': [0], 'addr_state_NM': [0], 'addr_state_NV': [0], 'addr_state_NY': [0], 'addr_state_OH': [0], 'addr_state_OK': [0], 'addr_state_OR': [0], 'addr_state_PA': [0], 'addr_state_RI': [0], 'addr_state_SC': [0], 'addr_state_SD': [0], 'addr_state_TN': [0], 'addr_state_TX': [0], 'addr_state_UT': [0], 'addr_state_VA': [0], 'addr_state_VT': [0], 'addr_state_WA': [0], 'addr_state_WI': [0], 'addr_state_WV': [0], 'addr_state_WY': [0]})

logreg.predict(new_data)


array([1])
