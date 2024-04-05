# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Perform data preprocessing steps such as encoding categorical variables, handling missing values, etc.
    # Split the data into features (X) and target variable (y)
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Perform feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train logistic regression model
def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Train random forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model

# Train gradient boosting model
def train_gradient_boosting(X_train, y_train):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    return gb_model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Yes')
    recall = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')
    return accuracy, precision, recall, f1

def main():
    # Load and preprocess the data
    data = load_data('data/dataset.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    # Evaluate models
    lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(lr_model, X_test, y_test)
    rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, X_test, y_test)
    gb_accuracy, gb_precision, gb_recall, gb_f1 = evaluate_model(gb_model, X_test, y_test)

    # Print results
    print("Logistic Regression Model:")
    print(f"Accuracy: {lr_accuracy}")
    print(f"Precision: {lr_precision}")
    print(f"Recall: {lr_recall}")
    print(f"F1 Score: {lr_f1}")
    print()
    print("Random Forest Model:")
    print(f"Accuracy: {rf_accuracy}")
    print(f"Precision: {rf_precision}")
    print(f"Recall: {rf_recall}")
    print(f"F1 Score: {rf_f1}")
    print()
    print("Gradient Boosting Model:")
    print(f"Accuracy: {gb_accuracy}")
    print(f"Precision: {gb_precision}")
    print(f"Recall: {gb_recall}")
    print(f"F1 Score: {gb_f1}")

if __name__ == "__main__":
    main()
