# ============================================================
# OCD-Early - 18Q Research Model Training
# ============================================================
# DEVELOPMENT / SYNTHETIC DATASET
#
# Primary ML inputs:
#   Age + Child_Q1..Child_Q18 + Parent_Q1..Parent_Q18
#
# Derived score fields are excluded from the primary experiment
# to reduce possible target leakage.
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "OCD_Early_Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\n================================================")
print("OCD-Early 18Q Research Model Training")
print("================================================")

print("\nDataset shape:", df.shape)

print("\nRisk Level Distribution:")
print(df["Risk_Level"].value_counts())


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

child_cols = [f"Child_Q{i}" for i in range(1, 19)]
parent_cols = [f"Parent_Q{i}" for i in range(1, 19)]

feature_columns = ["Age"] + child_cols + parent_cols
required_columns = feature_columns + ["Risk_Level"]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# 3. CLEAN DATA
# ============================================================

df = df.dropna(subset=required_columns).copy()

print(
    "\nDataset after required-value cleaning:",
    df.shape
)


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

X = df[feature_columns]
y = df["Risk_Level"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{i} = {label}")


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# 70% Training
# 15% Validation
# 15% Testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    stratify=y_encoded,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# 6. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=3.0,
        eval_metric="mlogloss",
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=15,
            weights="distance"
        ))
    ])
}


# ============================================================
# 7. MODEL TRAINING AND EVALUATION
# ============================================================

results = []
trained_models = {}
test_predictions = {}
cv_results = {}

print("\n================================================")
print("MODEL PERFORMANCE")
print("================================================")

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Store model and test predictions
    trained_models[name] = model
    test_predictions[name] = test_pred

    # Accuracy
    train_acc = accuracy_score(
        y_train,
        train_pred
    )

    val_acc = accuracy_score(
        y_val,
        val_pred
    )

    test_acc = accuracy_score(
        y_test,
        test_pred
    )

    # Generalisation gap
    generalisation_gap = train_acc - test_acc

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(
        y_test,
        test_pred
    )

    # Macro precision, recall and F1
    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            test_pred,
            average="macro",
            zero_division=0
        )
    )

    # Display model results
    print("\n--------------------------------------------")
    print("Model:", name)
    print("--------------------------------------------")

    print(
        "Training Accuracy:   ",
        round(train_acc, 4)
    )

    print(
        "Validation Accuracy: ",
        round(val_acc, 4)
    )

    print(
        "Test Accuracy:       ",
        round(test_acc, 4)
    )

    print(
        "Balanced Accuracy:   ",
        round(balanced_acc, 4)
    )

    print(
        "Macro Precision:      ",
        round(precision, 4)
    )

    print(
        "Macro Recall:         ",
        round(recall, 4)
    )

    print(
        "Macro F1:             ",
        round(macro_f1, 4)
    )

    print(
        "Generalisation Gap:   ",
        round(generalisation_gap, 4)
    )

    # Classification report
    print("\nTest Classification Report:")

    print(
        classification_report(
            y_test,
            test_pred,
            labels=np.arange(
                len(label_encoder.classes_)
            ),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Store results
    results.append({
        "Model": name,
        "Training Accuracy": train_acc,
        "Validation Accuracy": val_acc,
        "Test Accuracy": test_acc,
        "Balanced Accuracy": balanced_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Generalisation Gap": generalisation_gap
    })


# ============================================================
# 8. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n================================================")
print("5-FOLD CROSS VALIDATION")
print("================================================")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_results[name] = scores

    print("\nModel:", name)
    print(
        "Fold Scores:",
        np.round(scores, 4)
    )

    print(
        "Mean:",
        round(scores.mean(), 4)
    )

    print(
        "Std:",
        round(scores.std(), 4)
    )


# ============================================================
# 9. FINAL MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(results)

comparison["5-Fold CV Mean"] = [
    cv_results[name].mean()
    for name in comparison["Model"]
]

comparison["5-Fold CV Std"] = [
    cv_results[name].std()
    for name in comparison["Model"]
]

print("\n================================================")
print("FINAL MODEL COMPARISON")
print("================================================")

print(
    comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 10. BEST MODEL SELECTION
# ============================================================
# The best model is selected using the highest
# 5-fold cross-validation mean.
#
# Test accuracy is reported separately and is NOT
# used to select the best model.

best_model_name = max(
    cv_results,
    key=lambda name: cv_results[name].mean()
)

best_model = trained_models[best_model_name]

best_cv_accuracy = (
    cv_results[best_model_name].mean()
)

best_test_accuracy = comparison.loc[
    comparison["Model"] == best_model_name,
    "Test Accuracy"
].iloc[0]

best_macro_f1 = comparison.loc[
    comparison["Model"] == best_model_name,
    "Macro F1"
].iloc[0]


print("\n================================================")
print("BEST MODEL")
print("================================================")

print(
    "Best Model:",
    best_model_name
)

print(
    "Best CV Accuracy:",
    round(best_cv_accuracy, 4)
)

print(
    "Best Test Accuracy:",
    round(best_test_accuracy, 4)
)

print(
    "Best Macro F1:",
    round(best_macro_f1, 4)
)


# ============================================================
# 11. SAVE BEST MODEL AND RESULTS
# ============================================================

joblib.dump(
    best_model,
    "best_ocd_model.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

comparison.to_csv(
    "model_comparison_results.csv",
    index=False
)


# ============================================================
# 12. ALL CONFUSION MATRICES - ONE IMAGE
# ============================================================
# The four confusion matrices are displayed together
# in one 2x2 figure.
#
# Confusion-matrix numbers are NOT printed in the terminal.

class_names = label_encoder.classes_

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.ravel()

for i, name in enumerate(models.keys()):

    cm = confusion_matrix(
        y_test,
        test_predictions[name],
        labels=np.arange(
            len(class_names)
        )
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[i],
        cbar=False
    )

    axes[i].set_title(
        f"{name} Confusion Matrix",
        fontsize=13
    )

    axes[i].set_xlabel(
        "Predicted Class"
    )

    axes[i].set_ylabel(
        "Actual Class"
    )


fig.suptitle(
    "OCD-Early - Confusion Matrices of All ML Models",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    "all_models_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. FINAL OUTPUT FILES
# ============================================================

print("\n================================================")
print("FILES SAVED")
print("================================================")

print("best_ocd_model.pkl")
print("label_encoder.pkl")
print("model_comparison_results.csv")
print("all_models_confusion_matrices.png")

print("\nTraining completed successfully.")

