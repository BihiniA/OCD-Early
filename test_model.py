# ============================================================
# OCD-Early - 18Q ML Test / Prediction System
# ============================================================
# Uses the trained model:
#   best_ocd_model.pkl
#   label_encoder.pkl
#
# INPUTS USED BY THE TRAINED MODEL:
#   Age
#   Child_Q1 ... Child_Q18
#   Parent_Q1 ... Parent_Q18
#
# IMPORTANT:
# This is an AI screening/research prototype.
# It does not provide a clinical diagnosis.
# ============================================================

import joblib
import pandas as pd


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = joblib.load("best_ocd_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ============================================================
# 2. SYSTEM HEADER
# ============================================================

print("\n================================================")
print("        OCD-Early AI Screening System")
print("     Dual Parent–Adolescent OCD Screening")
print("================================================\n")

print("Please answer each question based on the")
print("child/adolescent's experiences during the")
print("last two weeks.\n")

print("Rating Scale:")
print("0 = Never")
print("1 = Rarely")
print("2 = Sometimes")
print("3 = Often")
print("4 = Very Often")


# ============================================================
# 3. INPUT VALIDATION FUNCTIONS
# ============================================================

def ask_age():
    while True:
        try:
            age = int(input("\nEnter child's age (10–18): "))

            if 10 <= age <= 18:
                return age

            print("Please enter an age between 10 and 18.")

        except ValueError:
            print("Invalid input. Please enter a number.")


def ask_question(question):
    while True:
        try:
            value = int(input(f"{question} (0–4): "))

            if 0 <= value <= 4:
                return value

            print("Please enter a number between 0 and 4.")

        except ValueError:
            print("Invalid input. Please enter a number from 0 to 4.")


# ============================================================
# 4. AGE INPUT
# ============================================================

age = ask_age()


# ============================================================
# 5. CHILD / ADOLESCENT QUESTIONS – 18
# ============================================================

print("\n================================================")
print("       CHILD / ADOLESCENT QUESTIONS")
print("================================================")

child_questions = [

    "1. Unwanted intrusive thoughts",

    "2. Fear of accidentally harming others",

    "3. Distressing sexual or inappropriate thoughts",

    "4. Repeated worries about relationships",

    "5. Religious or moral guilt",

    "6. Health-related worries",

    "7. Repeated checking behaviors",

    "8. Fear of contamination or excessive washing",

    "9. Need for symmetry or perfect order",

    "10. Mental rituals such as counting, praying, or analysing",

    "11. Avoiding triggers or situations",

    "12. Distress when rituals are interrupted",

    "13. Difficulty discarding items",

    "14. Thoughts or behaviors taking too much time",

    "15. Repeating thoughts or actions until they feel right",

    "16. Feeling responsible for preventing bad events",

    "17. Difficulty controlling obsessive thoughts or repetitive behaviors",

    "18. Interference with school, friendships, or daily activities"
]

child_answers = [
    ask_question(question)
    for question in child_questions
]


# ============================================================
# 6. PARENT QUESTIONS – 18
# ============================================================

print("\n================================================")
print("       PARENT QUESTIONS ABOUT THE CHILD")
print("================================================")

parent_questions = [

    "1. Repeated unwanted thoughts",

    "2. Fear of harming others",

    "3. Distress caused by inappropriate thoughts",

    "4. Repeated relationship worries",

    "5. Moral or religious anxiety",

    "6. Excessive health worries",

    "7. Repeated checking behaviors",

    "8. Excessive washing or contamination fears",

    "9. Need for perfect order or symmetry",

    "10. Repeating actions or mental rituals",

    "11. Avoidance of distressing situations",

    "12. Upset when rituals are interrupted",

    "13. Difficulty discarding items",

    "14. Excessive time spent on repetitive behaviors",

    "15. Repeating behaviors until they feel correct",

    "16. Excessive responsibility or fear of causing harm",

    "17. Difficulty controlling repetitive thoughts or behaviors",

    "18. Interference with school, family, or social life"
]

parent_answers = [
    ask_question(question)
    for question in parent_questions
]


# ============================================================
# 7. BUILD INPUT DATA
# ============================================================
# The feature order MUST exactly match the training script:
#
# Age
# Child_Q1 ... Child_Q18
# Parent_Q1 ... Parent_Q18
#
# Total = 37 features
# ============================================================

feature_columns = (
    ["Age"]
    + [f"Child_Q{i}" for i in range(1, 19)]
    + [f"Parent_Q{i}" for i in range(1, 19)]
)

input_data = (
    [age]
    + child_answers
    + parent_answers
)


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

df_input = pd.DataFrame(
    [input_data],
    columns=feature_columns
)


# ============================================================
# 9. ML MODEL PREDICTION
# ============================================================

prediction_encoded = model.predict(df_input)[0]

prediction_label = label_encoder.inverse_transform(
    [prediction_encoded]
)[0]


# ============================================================
# 10. DISPLAY INPUT SUMMARY
# ============================================================

child_total = sum(child_answers)
parent_total = sum(parent_answers)

print("\n================================================")
print("              SCREENING SUMMARY")
print("================================================")

print(f"\nChild Age: {age}")

print(f"Child Response Score:  {child_total}/72")
print(f"Parent Response Score: {parent_total}/72")


# ============================================================
# 11. DISPLAY AI PREDICTION
# ============================================================

print("\n================================================")
print("              AI PREDICTION")
print("================================================")

print(f"\nPredicted Risk Level: {prediction_label}")


# ============================================================
# 12. RISK-LEVEL INFORMATION
# ============================================================

print("\n================================================")
print("              SCREENING INFORMATION")
print("================================================")

if prediction_label == "Low":

    print("""
Low Risk Screening Result

The model classified the responses as Low Risk.

Continue monitoring the child's wellbeing and
maintain healthy routines and family support.
""")

elif prediction_label == "Moderate":

    print("""
Moderate Risk Screening Result

The model classified the responses as Moderate Risk.

The responses may indicate noticeable symptoms.
Consider discussing the concerns with an appropriate
mental-health professional if symptoms continue or
interfere with daily activities.
""")

elif prediction_label == "High":

    print("""
High Risk Screening Result

The model classified the responses as High Risk.

The responses indicate a higher level of screening
concern. Professional assessment by a qualified
mental-health professional is recommended.
""")

else:

    print("""
The model returned an unexpected risk category.
Please verify the trained model and label encoder.
""")


# ============================================================
# 13. FINAL NOTE
# ============================================================

print("\n================================================")
print("NOTE")
print("================================================")
print("OCD-Early is a research screening prototype.")
print("The AI prediction is not a clinical diagnosis.")
print("Professional assessment is required for diagnosis.")
print("================================================\n")