# OCD-Early: A Pre-Identification System for Obsessive–Compulsive Disorder (OCD) in Sri Lankan Adolescents

## 1. Project Overview

**OCD-Early** is a research-oriented mobile application designed to support early awareness of obsessive–compulsive disorder (OCD)-related symptoms among Sri Lankan adolescents aged 10–18 years.

The system combines adolescent self-reports and parent/guardian responses with machine learning to generate an indicative risk category and provide appropriate guidance for seeking further support.

OCD-Early is intended to support preliminary awareness and pre-identification, not to diagnose OCD or replace assessment by qualified mental health professionals.

## 2. Project Objectives

- Develop a machine learning-based approach for identifying indicative OCD-related risk patterns.
- Incorporate responses from both adolescents and their parents or guardians.
- Evaluate and compare multiple machine learning algorithms.
- Integrate the selected model with a Python Flask backend.
- Support communication between the backend and a Flutter mobile application.
- Provide appropriate recommendations based on the system's risk categorisation.

## 3. System Architecture

The system is designed around the following components:

1. **Mobile application:** Flutter-based user interface for collecting responses and displaying results.
2. **Backend API:** Python Flask application for receiving requests and returning predictions.
3. **Machine learning component:** Trained classification model for estimating the indicative risk category.
4. **Data processing component:** Preprocessing and feature preparation required for model inference.
5. **Result presentation:** Displays the predicted risk category and relevant guidance.

## 4. Technologies Used

- **Programming language:** Python, Dart
- **Machine learning:** Scikit-learn and other libraries used in model development
- **Backend:** Flask
- **Mobile application:** Flutter
- **API testing:** Postman
- **Development tools:** Visual Studio Code, Android Studio
- **Version control:** Git and GitHub

## 5. Machine Learning Models

The research evaluates the following classification algorithms:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- K-Nearest Neighbours (KNN)

The models are compared using classification performance metrics, including accuracy, precision, recall, F1-score, and cross-validation results.

The final model selection is based on the experimental evaluation and suitability for integration into the application.

## 6. Project Files

| File | Description |
|---|---|
| `app.py` | Flask backend API |
| `train_model.py` | Model training and evaluation script |
| `test_model.py` | Model testing script |
| `best_ocd_model.pkl` | Saved trained model, if included |
| `label_encoder.pkl` | Saved label encoder, if required |
| `model_comparison_results.csv` | Model comparison results, if included |
| `all_models_confusion_matrices.png` | Confusion matrix visualisation, if included |
| `OCD_Early_Dataset.csv` | Dataset used for model development; publish only when authorised |
| `README.md` | Project documentation |

## 7. Installation and Setup

### Prerequisites

Install the following software before running the backend:

- Python 3.10 or a compatible version supported by the project's dependencies
- Git
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/BihiniA/OCD-Early.git
cd OCD-Early
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows Git Bash:

```bash
source venv/Scripts/activate
```

Alternatively, in Windows Command Prompt:

```cmd
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the libraries required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

Otherwise, install the required libraries individually according to the imports and versions used in the project.

### Step 4: Run the Flask Backend

```bash
python app.py
```

The server address and port depend on the configuration in `app.py`. Check the terminal output for the actual address.

### Step 5: Test the API

Use Postman to send a request to the prediction endpoint configured in `app.py`. Supply the input fields and data format expected by that endpoint.

For mobile-device testing, ensure the phone and computer can communicate over the network, and configure the Flutter application to use the computer's reachable local IP address rather than `localhost`.

## 8. Model Evaluation

The research evaluates model performance using a training, validation, and independent test workflow, together with stratified cross-validation.

Reported results should be based on the actual experimental outputs. Performance on the test dataset does not, by itself, establish clinical effectiveness or diagnostic accuracy in real-world clinical settings.

## 9. Ethical Considerations and Limitations

- OCD-Early is a research prototype for preliminary risk awareness.
- It is not a diagnostic tool and must not be used as a substitute for professional assessment.
- Results should be interpreted cautiously and should not be used to label or stigmatise adolescents.
- Any real participant data must be handled according to applicable ethical approvals, consent requirements, privacy obligations, and data-sharing permissions.
- Synthetic data must be clearly distinguished from real-world participant data.
- Further validation using appropriate, ethically collected real-world data is required before considering clinical use.

## 10. Future Work

Potential future improvements include:

- Completing and refining the Flutter mobile application.
- Improving backend security and API validation.
- Conducting further evaluation with appropriately authorised real-world data.
- Improving accessibility and age-appropriate user guidance.
- Evaluating usability and technical performance.
- Conducting additional validation before any clinical application is considered.

## 11. Research Context

This project is being developed as part of an MSc research project in Information Technology at the Sri Lanka Institute of Information Technology (SLIIT).

**Research title:** *OCD-Early: A Pre-Identification System for Obsessive–Compulsive Disorder (OCD) in Sri Lankan Adolescents.*

---

**Disclaimer:** OCD-Early is a research prototype intended to support preliminary awareness. It does not provide a medical diagnosis, and its predictions should not be interpreted as confirmation that an individual has OCD.