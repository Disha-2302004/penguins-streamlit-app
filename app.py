#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('cleaned_penguins_size_processed.csv')

# Preprocess the data: Convert categorical variables to numerical
label_encoders = {}
for column in ['species', 'island', 'sex']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df.drop('species', axis=1)
y = df['species']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Check the balance
print(pd.Series(y_res).value_counts())

# Convert the balanced data back to a DataFrame
balanced_df = pd.DataFrame(X_res, columns=X.columns)
balanced_df['species'] = y_res

# If needed, convert the encoded values back to their original categorical values
for column in ['species', 'island', 'sex']:
    balanced_df[column] = label_encoders[column].inverse_transform(balanced_df[column])

# Save the balanced dataset to a new CSV file
balanced_df.to_csv('balanced_penguins_size_processed.csv', index=False)


# In[27]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 


# In[28]:


file_path = "E:/research/balanced_penguins_size_processed.csv"  
data = pd.read_csv(file_path)


# In[29]:


print("Dataset Preview:")
print(data.head())


# In[30]:


data['sex'] = data['sex'].replace('.', np.nan)


# In[31]:


print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())


# In[32]:


data_cleaned = data.dropna()


# In[33]:


print("\nSummary Statistics:")
print(data_cleaned.describe())


# In[9]:


sns.set(style="whitegrid")


# In[10]:


# 1. Bar plot: Distribution of species
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.countplot(data=data_cleaned, x='species', palette='Set2', hue='species', dodge=False, legend=False)
plt.title('Distribution of Penguin Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()


# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv" 
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),  
    "LightGBM": LGBMClassifier(min_child_samples=10, min_split_gain=0.01, max_depth=10, verbose=-1),  
    "CatBoost": CatBoostClassifier(verbose=0),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} - Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%, Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=1)* 100:.2f}%, Recall: {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%, F1 Score: {f1_score(y_test, y_pred, average='weighted')* 100:.2f}%")
    print(f"Confusion Matrix for {name}:\n{confusion_matrix(y_test, y_pred)}\n")


# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv" 
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into 70% training and 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Split remaining 30% into 20% test and 10% validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Print dataset sizes
print(f"Training set size: {len(y_train)} ({len(y_train)/len(y):.0%})")
print(f"Testing set size: {len(y_test)} ({len(y_test)/len(y):.0%})")
print(f"Validation set size: {len(y_val)} ({len(y_val)/len(y):.0%})\n")

# Define classifiers
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)  # Train on 70% dataset

    # Predictions on all datasets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)

    # Combine all predictions
    y_true_combined = np.concatenate([y_train, y_test, y_val])
    y_pred_combined = np.concatenate([y_train_pred, y_test_pred, y_val_pred])

    # Compute combined evaluation metrics
    accuracy = accuracy_score(y_true_combined, y_pred_combined)
    precision = precision_score(y_true_combined, y_pred_combined, average='weighted', zero_division=1)
    recall = recall_score(y_true_combined, y_pred_combined, average='weighted')
    f1 = f1_score(y_true_combined, y_pred_combined, average='weighted')

    # Compute combined confusion matrix
    combined_cm = confusion_matrix(y_true_combined, y_pred_combined)

    # Print evaluation results
    print(f"\n=== {name} Model  ===")
    print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")
    print(f"\nCombined Confusion Matrix:\n{combined_cm}\n")


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv" 
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into 70% training, 20% testing, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Define classifiers
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Initialize lists to store metrics
train_accuracies, test_accuracies, log_losses = {}, {}, {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)  # Train on 70% dataset

    # Predictions on training & testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Store results
    train_accuracies[name] = train_acc
    test_accuracies[name] = test_acc

    # Compute log loss (if model supports probability outputs)
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_test_proba)
    else:
        loss = np.nan  # Not applicable for some models

    log_losses[name] = loss

# Function to add value labels on bars
def add_value_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot Training & Testing Accuracy
plt.figure(figsize=(14, 5))

# Training Accuracy
plt.subplot(1, 2, 1)
sns.barplot(
    x=list(train_accuracies.keys()), 
    y=list(train_accuracies.values()), 
    hue=list(train_accuracies.keys()),  
    palette="Blues_r", 
    legend=False  
)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Training Accuracy of Models")
add_value_labels(ax1) 

# Testing Accuracy
plt.subplot(1, 2, 2)
sns.barplot(
    x=list(test_accuracies.keys()), 
    y=list(test_accuracies.values()), 
    hue=list(test_accuracies.keys()),  
    palette="Greens_r", 
    legend=False  
)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Testing Accuracy of Models")
add_value_labels(ax2)

plt.tight_layout()
plt.show()

# Plot Log Loss for models that support probabilities
valid_log_losses = {k: v for k, v in log_losses.items() if not np.isnan(v)}

plt.figure(figsize=(10, 5))
sns.barplot(
    x=list(test_accuracies.keys()), 
    y=list(test_accuracies.values()), 
    hue=list(test_accuracies.keys()),  
    palette="Reds_r", 
    legend=False  
)
plt.xticks(rotation=90)
plt.ylabel("Log Loss")
plt.title("Loss (Log Loss) of Models")
add_value_labels(ax3) 
plt.show()





# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders['species'].classes_, yticklabels=label_encoders['species'].classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name} (Full Dataset)")
    plt.show()

# Combine training, testing, and validation data
X_full = np.vstack((X_train, X_test, X_val))  # Merge all features
y_full = np.hstack((y_train, y_test, y_val))  # Merge all labels

# Generate confusion matrices for all models using full dataset
for name, model in models.items():
    y_pred_full = model.predict(X_full)  # Predict on full dataset
    plot_confusion_matrix(y_full, y_pred_full, f"{name}")


# In[ ]:


xgb_model = XGBClassifier(eval_metric="mlogloss")
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

# Extract and plot loss
loss_train = xgb_model.evals_result_["validation_0"]["mlogloss"]
loss_val = xgb_model.evals_result_["validation_1"]["mlogloss"]

plt.plot(loss_train, label="Train Loss")
plt.plot(loss_val, label="Validation Loss")
plt.title("XGBoost Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Extract loss values correctly
loss_train = lgb_model.evals_result_["training"]["multi_logloss"]
loss_val = lgb_model.evals_result_["valid_1"]["multi_logloss"]  # Fix the key

# Plot training & validation loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_train, label="Training Loss")
plt.plot(loss_val, label="Validation Loss", linestyle="dashed")
plt.xlabel("Iterations")
plt.ylabel("Multi Log Loss")
plt.title("LightGBM Loss Curve")
plt.legend()
plt.show()






# In[ ]:


# Extract loss values correctly
loss_train = cat_model.get_evals_result()["learn"]["MultiClass"]
loss_val = cat_model.get_evals_result()["validation_0"]["MultiClass"]  # Correct key

# Plot loss curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(loss_train, label="Training Loss")
plt.plot(loss_val, label="Validation Loss", linestyle="dashed")
plt.xlabel("Iterations")
plt.ylabel("Multi-Class Loss")
plt.title("CatBoost Loss Curve")
plt.legend()
plt.show()



# In[14]:


mlp_model = MLPClassifier(max_iter=500)
mlp_model.fit(X_train, y_train)

# Extract loss
plt.plot(mlp_model.loss_curve_, label="Train Loss")
plt.title("MLP Neural Network Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Function to add value labels on bars
def add_value_labels(ax, spacing=5):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Avoid placing labels on zero-height bars
            ax.annotate(
                f'{height:.3f}',  # Format to 3 decimal places
                (p.get_x() + p.get_width() / 2., height + spacing * 0.01),  # Adjust spacing dynamically
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
            )

# Set the figure size
plt.figure(figsize=(10, 6))

# Training Accuracy Plot
ax1 = sns.barplot(
    x=list(train_accuracies.keys()), 
    y=list(train_accuracies.values()), 
    palette="Blues_r"
)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)  # Increase limit slightly for better label visibility
plt.title("Training Accuracy of Models")
add_value_labels(ax1)
plt.show()  # Display first plot

# Testing Accuracy Plot
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(
    x=list(test_accuracies.keys()), 
    y=list(test_accuracies.values()), 
    palette="Greens_r"
)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)  # Increase limit slightly for better label visibility
plt.title("Testing Accuracy of Models")
add_value_labels(ax2)
plt.show()  # Display second plot



# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Filter out NaN values for valid log loss models
valid_log_losses = {k: v for k, v in log_losses.items() if not np.isnan(v)}

# Sort log losses for better visualization
sorted_log_losses = dict(sorted(valid_log_losses.items(), key=lambda item: item[1], reverse=True))

# Create the bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=list(sorted_log_losses.keys()), 
    y=list(sorted_log_losses.values()), 
    palette="Reds_r"
)

# Label each bar with the corresponding log loss value
for i, v in enumerate(sorted_log_losses.values()):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')

# Graph formatting
plt.xticks(rotation=90)
plt.ylabel("Log Loss")
plt.ylim(0, max(sorted_log_losses.values()) * 1.1)  # Adjust y-axis dynamically
plt.title("Loss (Log Loss) of Models")

plt.show()


# In[18]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Binarize the output labels for multi-class classification
y_train_bin = label_binarize(y_train, classes=np.unique(y))
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_val_bin = label_binarize(y_val, classes=np.unique(y))
n_classes = y_train_bin.shape[1]

plt.figure(figsize=(12, 8))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'])

# Plot ROC curve for each model
for (name, model), color in zip(models.items(), colors):
    if hasattr(model, "predict_proba"):  # Check if model supports predict_proba
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot random classifier line
plt.plot([0, 1], [0, 1], linestyle='--', color='black', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.5), fontsize='small')  # Adjust legend size and position
plt.show()


# In[21]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the output labels for multi-class classification
y_train_bin = label_binarize(y_train, classes=np.unique(y))
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_train_bin.shape[1]

# Find the best performing model based on AUC
best_model = None
best_auc = 0
best_model_name = ""

for name, model in models.items():
    if hasattr(model, "predict_proba"):  # Check if model supports predict_proba
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
    
    # Compute mean AUC for all classes
    auc_scores = [auc(*roc_curve(y_test_bin[:, i], y_score[:, i])[:2]) for i in range(n_classes)]
    mean_auc = np.mean(auc_scores)
    
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model
        best_model_name = name

# Define colors ensuring orange is visible
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plot ROC curve for the best performing model
plt.figure(figsize=(10, 6))
y_score = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# Plot random classifier line
plt.plot([0, 1], [0, 1], linestyle='--', color='black', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Best Model: {best_model_name}')
plt.legend(loc='lower right', fontsize='small')
plt.show()


# In[43]:


pip install --upgrade xgboost


# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Reduce dataset size for tuning
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.75, random_state=42)

# Adjusted Hyperparameter tuning settings
tuned_models = {}
param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "HistGradientBoosting": {
        'max_iter': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "SVC": {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance']
    },
    "NaiveBayes": {},
    "DecisionTree": {
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    },
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,)],
        'alpha': [0.0001, 0.001]
    }
}

# Perform hyperparameter tuning
for name, param_grid in param_grids.items():
    model_cls = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }[name]
    
    if param_grid:
        search = RandomizedSearchCV(model_cls, param_grid, cv=5, n_iter=min(3, len(list(GridSearchCV(model_cls, param_grid).param_grid))), scoring='accuracy', n_jobs=1, random_state=42)
        search.fit(X_train_sample, y_train_sample)
        tuned_models[name] = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        tuned_models[name] = model_cls.fit(X_train_sample, y_train_sample)


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Reduce dataset size for tuning
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.75, random_state=42)

# Adjusted Hyperparameter tuning settings
tuned_models = {}
param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "HistGradientBoosting": {
        'max_iter': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "SVC": {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance']
    },
    "NaiveBayes": {},
    "DecisionTree": {
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    },
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,)],
        'alpha': [0.0001, 0.001]
    }
}

# Perform hyperparameter tuning
for name, param_grid in param_grids.items():
    model_cls = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }[name]
    
    if param_grid:
        search = RandomizedSearchCV(model_cls, param_grid, cv=5, n_iter=min(3, len(list(GridSearchCV(model_cls, param_grid).param_grid))), scoring='accuracy', n_jobs=1, random_state=42)
        search.fit(X_train_sample, y_train_sample)
        tuned_models[name] = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        tuned_models[name] = model_cls.fit(X_train_sample, y_train_sample)

# Evaluate models
model_performance = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")


# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Adjusted Hyperparameter tuning settings
tuned_models = {}
param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "HistGradientBoosting": {
        'max_iter': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "SVC": {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance']
    },
    "NaiveBayes": {},
    "DecisionTree": {
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    },
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,)],
        'alpha': [0.0001, 0.001]
    }
}

# Perform hyperparameter tuning
for name, param_grid in param_grids.items():
    model_cls = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }[name]
    
    if param_grid:
        search = RandomizedSearchCV(model_cls, param_grid, cv=5, n_iter=min(3, len(list(GridSearchCV(model_cls, param_grid).param_grid))), scoring='accuracy', n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        tuned_models[name] = model_cls.fit(X_train, y_train)

# Evaluate models
model_performance = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Adjusted Hyperparameter tuning settings
tuned_models = {}
param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "HistGradientBoosting": {
        'max_iter': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "SVC": {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance']
    },
    "NaiveBayes": {},
    "DecisionTree": {
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    },
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,)],
        'alpha': [0.0001, 0.001]
    }
}

# Perform hyperparameter tuning
for name, param_grid in param_grids.items():
    model_cls = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }[name]
    
    if param_grid:
        search = RandomizedSearchCV(model_cls, param_grid, cv=5, n_iter=min(3, len(list(GridSearchCV(model_cls, param_grid).param_grid))), scoring='accuracy', n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        tuned_models[name] = model_cls.fit(X_train, y_train)

# Evaluate models
model_performance = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "E:/research/balanced_penguins_size_processed.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['species', 'island', 'sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['species'])
y = df['species']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Use all 456 records for training
X_train, X_test, y_train, y_test = X, X, y, y

# Adjusted Hyperparameter tuning settings
tuned_models = {}
param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "HistGradientBoosting": {
        'max_iter': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1]
    },
    "SVC": {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance']
    },
    "NaiveBayes": {},
    "DecisionTree": {
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    },
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,)],
        'alpha': [0.0001, 0.001]
    }
}

# Perform hyperparameter tuning
for name, param_grid in param_grids.items():
    model_cls = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }[name]
    
    if param_grid:
        search = RandomizedSearchCV(model_cls, param_grid, cv=5, n_iter=min(3, len(list(GridSearchCV(model_cls, param_grid).param_grid))), scoring='accuracy', n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        tuned_models[name] = model_cls.fit(X_train, y_train)

# Evaluate models
model_performance = {}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()


# In[1]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')


# In[ ]:




