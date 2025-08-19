# AI-powered-Network-Intrusion-Detection

This project implements a network intrusion detection system using a Support Vector Machine (SVM) classifier. The model is trained on the NSL-KDD dataset to perform multi-class classification, identifying different types of network attacks as well as normal traffic.

## Dataset

The model is trained on the [NSL-KDD dataset](https://www.kaggle.com/datasets/hassan06/nslkdd/data), which is an improved version of the original KDD'99 dataset. It is designed to solve some of the inherent problems of the KDD'99 dataset and is widely used for network intrusion detection research.

The dataset contains various types of network intrusions, which can be broadly categorized into the following groups:
*   **Denial of Service (DoS):** Attacks that make a machine or network resource unavailable to its intended users.
*   **Probe:** Attacks that gather information about a network or a computer system.
*   **Remote to Local (R2L):** Attacks in which an attacker sends packets to a machine over a network, then exploits a vulnerability to gain local access.
*   **User to Root (U2R):** Attacks in which an attacker starts out with access to a normal user account on the system and is able to exploit a vulnerability to gain root access.

## Code Overview

The `main.py` script performs the following steps:

1.  **Data Loading:** It loads the NSL-KDD training and testing datasets (`KDDTrain+.txt` and `KDDTest+.txt`) using Pandas.
2.  **Data Cleaning:** It drops the `difficulty_level` and `num_outbound_cmds` columns from the datasets.
3.  **Feature Engineering:**
    *   It encodes categorical features (`protocol_type`, `service`, `flag`) into numerical representations using `LabelEncoder`.
    *   It encodes the multi-class `label` for both training and testing sets.
4.  **Data Splitting and Scaling:**
    *   The training data is split into a training set (70%) and a validation set (30%).
    *   Features are standardized using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1.
5.  **Model Training:**
    *   An `SVC` (Support Vector Classifier) with an RBF kernel is trained on the scaled training data.
    *   The model's performance is evaluated on the validation set.
6.  **Model Persistence:** The trained model is saved to `SVM_model_multiclass.pkl` using `pickle`.
7.  **Final Evaluation:** The model is evaluated on the independent test set to assess its generalization performance on unseen data.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI-powered-Network-Intrusion-Detection.git
    cd AI-powered-Network-Intrusion-Detection
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 and pip installed. Then, install the required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute the script:**
    Run the `main.py` script from your terminal:
    ```bash
    python main.py
    ```

## Expected Output

When you run the script, it will print the following to the console:

*   **Internal Validation Results:** The accuracy of the model on the internal validation set, followed by a detailed classification report.
*   **Model Save Confirmation:** A message confirming that the trained model has been saved.
*   **Final Test Results:** The accuracy of the model on the independent test set, followed by another detailed classification report.

The classification report will look similar to this:

```
              precision    recall  f1-score   support

      attack1       0.99      0.98      0.99      1000
      attack2       0.85      0.90      0.87       500
       normal       0.99      1.00      0.99     20000
          ...        ...       ...       ...       ...

     accuracy                           0.98     21500
    macro avg       0.94      0.96      0.95     21500
 weighted avg       0.98      0.98      0.98     21500
```

## References

This code is a modified version of the implementation from the book "Artificial Intelligence for Cybersecurity" by Packt Publishing. The original code can be found here:
[https://github.com/PacktPublishing/Artificial-Intelligence-for-Cybersecurity/blob/main/Chapter%2007/Network%20Intrusion%20Detection/main.py](https://github.com/PacktPublishing/Artificial-Intelligence-for-Cybersecurity/blob/main/Chapter%2007/Network%20Intrusion%20Detection/main.py)
