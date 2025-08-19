# This code uses a trainied SVC system to detect attack types. 
# It is modified from https://github.com/PacktPublishing/Artificial-Intelligence-for-Cybersecurity/blob/main/Chapter%2007/Network%20Intrusion%20Detection/main.py for multi-class labelling.

import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Ignore future warnings to keep the output clean
warnings.filterwarnings('ignore')

# Define the column names for the NSL-KDD dataset.
# This tuple contains all 43 features, plus the 'label' and 'difficulty_level' columns.
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label',
    'difficulty_level'
]


# --- 1. Data Loading and Initial Cleaning ---

# Load the NSL-KDD training and testing datasets into Pandas DataFrames.
train_file_path = r'C:\Users\chenj\Documents\GitHub\Artificial-Intelligence-for-Cybersecurity\Chapter 07\Network Intrusion Detection\dataset\KDDTrain+.txt'
test_file_path = r'C:\Users\chenj\Documents\GitHub\Artificial-Intelligence-for-Cybersecurity\Chapter 07\Network Intrusion Detection\dataset\KDDTest+.txt'

train_data = pd.read_csv(train_file_path, header=None, names=COLUMNS)
test_data = pd.read_csv(test_file_path, header=None, names=COLUMNS)

# Drop 'difficulty_level' and 'num_outbound_cmds' columns.
train_data.drop(['difficulty_level', 'num_outbound_cmds'],
                axis=1,
                inplace=True)
test_data.drop(['difficulty_level', 'num_outbound_cmds'],
               axis=1,
               inplace=True)


# --- 2. Feature Engineering ---

# Encode categorical features: 'protocol_type', 'service', 'flag'
# Use LabelEncoder to convert string features into numerical labels.
# We fit the encoder on the combined data from both training and testing sets
# to ensure all possible categories are learned.
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    encoder = LabelEncoder()
    # Combine data from both train and test sets to learn all possible categories
    combined_data = pd.concat([train_data[col], test_data[col]]).unique()
    encoder.fit(combined_data)
    # Transform both train and test data using the fitted encoder
    train_data[col] = encoder.transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])

# Encode the multi-class target variable 'label'
# Combine labels from both train and test sets to ensure the encoder
# is aware of all possible classes before transforming.
combined_labels = pd.concat([train_data['label'], test_data['label']]).unique()
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)

train_data['label'] = label_encoder.transform(train_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])

# Save label names for use in the classification report later
label_names = label_encoder.classes_


# --- 3. Data Splitting and Feature Scaling ---

# Separate the training data into features (X) and the target variable (y).
X = train_data.drop(['label'], axis=1)
y = train_data['label']

# Split the data into a training set and a validation set for model evaluation.
# 30% of the data is used for validation. `random_state` ensures reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size=0.30,
                                                  random_state=40)

# Standardize features by removing the mean and scaling to unit variance.
# This is crucial for distance-based algorithms like SVM.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# --- 4. Model Training and Validation ---

# Initialize a Support Vector Classifier (SVC) with a Radial Basis Function (RBF) kernel.
print("Starting SVM model training (this may take some time)...")
svm_model = SVC(kernel='rbf')

# Train the model using the preprocessed and scaled training data.
svm_model.fit(X_train, y_train)

# Make predictions on the validation set.
y_pred_val = svm_model.predict(X_val)

# Evaluate the model's performance on the validation set.
print('--- Internal Validation Set Evaluation ---')
print(f'Training Accuracy: {svm_model.score(X_train, y_train):.4f}')
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}')
print('\nValidation Set Classification Report:\n')

# Add the `labels` parameter to handle rare classes that might be
# missing in the validation set, ensuring a complete report.
print(
    classification_report(y_val,
                          y_pred_val,
                          labels=np.arange(len(label_names)),
                          target_names=label_names))


# --- 5. Model Persistence ---

# Save the trained model to disk using pickle for future use.
pkl_filename = "SVM_model_multiclass.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svm_model, file)
print(f"Model saved to: {pkl_filename}")


# --- 6. Final Evaluation on Independent Test Set ---

print('\n\n--- Final Evaluation Using Independent Test Set ---')

# Prepare the test data.
X_test_final = test_data.drop(['label'], axis=1)
y_test_final = test_data['label']

# Apply the same scaling to the test set features that was learned from the training set.
X_test_final = scaler.transform(X_test_final)

# Use the trained model to make predictions on the final test set.
y_pred_final = svm_model.predict(X_test_final)

# Evaluate the model's generalization ability on unseen data.
print(f'Final Test Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}')
print('\nFinal Test Set Classification Report:\n')

# Add the `labels` parameter here as well for a complete report.
print(
    classification_report(y_test_final,
                          y_pred_final,
                          labels=np.arange(len(label_names)),
                          target_names=label_names))
