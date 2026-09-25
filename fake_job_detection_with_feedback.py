# Import required libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import shap                                          
import joblib

# Step 1: Load and preprocess data                             
def load_data(filepath):
    data = pd.read_csv(filepath)  # Columns: 'text', 'label'
    data['text'] = data['title'] + ' ' + data['location'] + ' ' + data['company_profile'] + ' ' + data['description'] + ' ' + data['requirements'] + ' ' + data['benefits']
    data.fillna(' ', inplace=True)  # Fill NaN values with blank space
    return data

# Step 2: Feature extraction and model training
def train_model(data):
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['fraudulent']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, vectorizer

# Step 3: Explainability with SHAP
def explain_prediction(model, vectorizer, text):
    explainer = shap.LinearExplainer(model, vectorizer.transform(data['text']))
    shap_values = explainer.shap_values(vectorizer.transform([text]))
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_values[0], feature_names=vectorizer.get_feature_names_out())

# Step 4: User feedback mechanism
def collect_feedback(prediction, text):
    print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    feedback = input("Was the prediction correct? (yes/no): ").strip().lower()
    if feedback == 'no':
        correct_label = int(input("Enter the correct label (0 for Real, 1 for Fake): "))
        return {'text': text, 'label': correct_label}
    return None

# Step 5: Real-time prediction with explainability and feedback
def predict_and_explain(model, vectorizer, text):
    # Predict
    input_data_features = vectorizer.transform([text])
    prediction = model.predict(input_data_features)[0]

    # Explain
    explanation = explain_prediction(model, vectorizer, text)

    # Collect feedback
    feedback = collect_feedback(prediction, text)
    return prediction, explanation, feedback

# Step 6: Update model with feedback
def update_model(model, vectorizer, feedback_data):
    new_texts = [feedback_data['text']]
    new_labels = [feedback_data['label']]
    new_X = vectorizer.transform(new_texts)
    model.partial_fit(new_X, new_labels, classes=[0, 1])

# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data('fake_job_postings.csv')  # Replace with your dataset path

    # Train model
    model, vectorizer = train_model(data)

    # Save model and vectorizer
    joblib.dump(model, 'fake_job_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Example prediction with explainability and feedback
    input_text = "Earn $5000 a week from home! No experience needed."
    prediction, explanation, feedback = predict_and_explain(model, vectorizer, input_text)

    # Update model with feedback if provided
    if feedback:
        update_model(model, vectorizer, feedback)
        print("Model updated with user feedback!")