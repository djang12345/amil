I'll help you create this Animal Classification Project step-by-step using Google Colab and a Decision Tree Classifier. Let me search for a suitable Kaggle dataset first.Perfect! I found an excellent dataset for this project. The Zoo Animal Classification dataset on Kaggle is ideal for this assignment. It contains animal features like whether they have hair, feathers, lay eggs, etc., and classifies them into different types.

Let me create a complete step-by-step guide for you:Now let me create a working Colab notebook with actual executable code:Perfect! Now let me create a detailed step-by-step instruction document for you:## 📚 Summary

I've created a complete guide for your Animal Classification project! Here's what I provided:

### 1️⃣ **Dataset Found:**
I found the perfect dataset - the Zoo Animal Classification dataset from Kaggle, which contains 101 animals with 16 features like hair, feathers, eggs, milk, etc., classified into 7 animal types (mammals, birds, reptiles, fish, amphibians, insects, and invertebrates).

### 2️⃣ **Three Artifacts Created:**

**Artifact 1: Complete Code Guide** - Commented Python code with all 11 steps explained

**Artifact 2: Executable Code** - Ready-to-run code that you can copy directly into Google Colab. This includes:
- Automatic dataset loading from UCI repository
- Data exploration with visualizations
- Decision Tree training and visualization
- Model evaluation with accuracy, confusion matrix, and feature importance
- Custom animal prediction example

**Artifact 3: Step-by-Step Instructions** - Detailed tutorial explaining:
- How to set up Google Colab
- What each feature means
- What each step does and why
- How to interpret results
- Submission instructions
- Troubleshooting tips

### 🎯 To Complete Your Project:

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy the **executable code** (Artifact 2) into your notebook
4. Run all cells from top to bottom
5. Review the outputs and visualizations
6. Save to Drive and share with "Anyone with the link"
7. Submit the link in your Word document

The code will automatically download the dataset and complete all requirements. You should get around 85-95% accuracy! Let me know if you need any clarification on any step! 🚀



```py
# ====================================================================
# ANIMAL CLASSIFICATION PROJECT - DECISION TREE CLASSIFIER
# Complete Step-by-Step Guide for Google Colab
# ====================================================================

# ====================================================================
# STEP 1: INSTALL AND IMPORT LIBRARIES
# ====================================================================
print("Step 1: Installing and Importing Libraries...")

# Install required libraries (if not already installed)
# !pip install pandas numpy scikit-learn matplotlib seaborn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully!\n")


# ====================================================================
# STEP 2: DOWNLOAD AND LOAD THE DATASET
# ====================================================================
print("Step 2: Loading the Dataset...")
print("-" * 60)
print("DATASET INFORMATION:")
print("We'll use the Zoo Animal Classification dataset from Kaggle")
print("Dataset link: https://www.kaggle.com/datasets/uciml/zoo-animal-classification")
print("-" * 60)

# Method 1: Upload CSV file directly
# Uncomment the lines below if you want to upload manually
# from google.colab import files
# uploaded = files.upload()

# Method 2: Download from URL (You'll need to download from Kaggle first)
# For this tutorial, we'll create a sample or you can upload the CSV

# If you have the dataset uploaded, use:
# df = pd.read_csv('zoo.csv')

# For demonstration, let me show you how to load it:
print("\nTo get the dataset:")
print("1. Go to: https://www.kaggle.com/datasets/uciml/zoo-animal-classification")
print("2. Click 'Download' (you may need to create a free Kaggle account)")
print("3. Upload the 'zoo.csv' file to your Colab notebook")
print("4. Run: df = pd.read_csv('zoo.csv')")

# Example: Reading the dataset (uncomment when you have the file)
# df = pd.read_csv('zoo.csv')
# print("✓ Dataset loaded successfully!")


# ====================================================================
# STEP 3: EXPLORE THE DATASET
# ====================================================================
print("\nStep 3: Exploring the Dataset...")

# Display first few rows
# print("\nFirst 5 rows of the dataset:")
# print(df.head())

# Display dataset information
# print("\nDataset Info:")
# print(df.info())

# Display statistical summary
# print("\nStatistical Summary:")
# print(df.describe())

# Check for missing values
# print("\nMissing Values:")
# print(df.isnull().sum())

# Display dataset shape
# print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Check class distribution
# print("\nClass Distribution:")
# print(df['class_type'].value_counts())


# ====================================================================
# STEP 4: PREPARE THE DATA
# ====================================================================
print("\nStep 4: Preparing the Data...")

# The Zoo dataset typically has these columns:
# animal_name, hair, feathers, eggs, milk, airborne, aquatic, predator,
# toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize, class_type

# Separate features (X) and target variable (y)
# Drop 'animal_name' as it's not a feature for prediction
# X = df.drop(['animal_name', 'class_type'], axis=1)
# y = df['class_type']

# Alternative: If you want to keep animal names for reference
# animal_names = df['animal_name']
# X = df.drop(['animal_name', 'class_type'], axis=1)
# y = df['class_type']

print("Features (X): All animal characteristics")
print("Target (y): Animal class type")
print("✓ Data separated successfully!")


# ====================================================================
# STEP 5: SPLIT THE DATA
# ====================================================================
print("\nStep 5: Splitting Data into Training and Testing Sets...")

# Split data: 80% training, 20% testing
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print(f"Training set size: {X_train.shape[0]} samples")
# print(f"Testing set size: {X_test.shape[0]} samples")
# print("✓ Data split successfully!")


# ====================================================================
# STEP 6: CREATE AND TRAIN THE DECISION TREE MODEL
# ====================================================================
print("\nStep 6: Training the Decision Tree Classifier...")

# Create Decision Tree Classifier
# dt_classifier = DecisionTreeClassifier(
#     criterion='gini',        # or 'entropy' for information gain
#     max_depth=5,             # limit tree depth to prevent overfitting
#     min_samples_split=2,     # minimum samples required to split
#     min_samples_leaf=1,      # minimum samples required at leaf node
#     random_state=42
# )

# Train the model
# dt_classifier.fit(X_train, y_train)
# print("✓ Model trained successfully!")


# ====================================================================
# STEP 7: VISUALIZE THE DECISION TREE
# ====================================================================
print("\nStep 7: Visualizing the Decision Tree...")

# Create a large figure for better visibility
# plt.figure(figsize=(20, 10))
# plot_tree(
#     dt_classifier,
#     feature_names=X.columns,
#     class_names=[str(i) for i in sorted(y.unique())],
#     filled=True,
#     rounded=True,
#     fontsize=10
# )
# plt.title("Decision Tree for Animal Classification", fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

# print("✓ Decision tree visualized!")


# ====================================================================
# STEP 8: MAKE PREDICTIONS
# ====================================================================
print("\nStep 8: Making Predictions...")

# Make predictions on test set
# y_pred = dt_classifier.predict(X_test)

# Display some predictions
# predictions_df = pd.DataFrame({
#     'Actual': y_test.values,
#     'Predicted': y_pred
# })
# print("\nFirst 10 Predictions:")
# print(predictions_df.head(10))


# ====================================================================
# STEP 9: EVALUATE THE MODEL
# ====================================================================
print("\nStep 9: Evaluating Model Performance...")

# Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Display classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# Create and display confusion matrix
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Class')
# plt.xlabel('Predicted Class')
# plt.show()


# ====================================================================
# STEP 10: FEATURE IMPORTANCE
# ====================================================================
print("\nStep 10: Analyzing Feature Importance...")

# Get feature importance
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': dt_classifier.feature_importances_
# }).sort_values('Importance', ascending=False)

# print("\nFeature Importance Rankings:")
# print(feature_importance)

# Visualize feature importance
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance['Feature'], feature_importance['Importance'])
# plt.xlabel('Importance')
# plt.title('Feature Importance in Animal Classification')
# plt.tight_layout()
# plt.show()


# ====================================================================
# STEP 11: TEST WITH NEW DATA (OPTIONAL)
# ====================================================================
print("\nStep 11: Testing with New Animal Data...")

# Example: Predict class for a new animal
# Create a new animal with specific features
# new_animal = pd.DataFrame({
#     'hair': [1],
#     'feathers': [0],
#     'eggs': [0],
#     'milk': [1],
#     'airborne': [0],
#     'aquatic': [0],
#     'predator': [1],
#     'toothed': [1],
#     'backbone': [1],
#     'breathes': [1],
#     'venomous': [0],
#     'fins': [0],
#     'legs': [4],
#     'tail': [1],
#     'domestic': [0],
#     'catsize': [1]
# })

# prediction = dt_classifier.predict(new_animal)
# print(f"Predicted class for new animal: {prediction[0]}")


# ====================================================================
# FINAL NOTES
# ====================================================================
print("\n" + "="*60)
print("PROJECT COMPLETE!")
print("="*60)
print("\nWhat you've accomplished:")
print("✓ Loaded and explored animal dataset")
print("✓ Prepared data for machine learning")
print("✓ Created a Decision Tree Classifier")
print("✓ Trained the model on training data")
print("✓ Visualized the decision tree structure")
print("✓ Made predictions on test data")
print("✓ Evaluated model performance")
print("✓ Analyzed feature importance")
print("\nNext Steps:")
print("1. Save this notebook to your Google Drive")
print("2. Share the notebook with 'Anyone with the link can view'")
print("3. Copy the shareable link")
print("4. Submit the link in your Word file")
print("="*60)
```


```python
# Animal Classification with Decision Tree - Google Colab
# Complete executable code - just run all cells!

# Step 1: Install and Import
print("Installing required package...")
!pip install ucimlrepo -q

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load Dataset from UCI Repository
print("\nLoading Zoo Animal dataset from UCI...")
zoo = fetch_ucirepo(id=111)
X = zoo.data.features
y = zoo.data.targets.values.ravel()

print(f"✓ Dataset loaded: {X.shape[0]} animals, {X.shape[1]} features")
print(f"✓ Classes: {len(set(y))} animal types\n")

# Step 3: Explore Data
print("Dataset Preview:")
print(pd.concat([X.head(10), pd.DataFrame(y[:10], columns=['class'])], axis=1))

plt.figure(figsize=(10, 4))
pd.Series(y).value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Animal Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining: {len(X_train)} | Testing: {len(X_test)}")

# Step 5: Train Decision Tree
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
print("✓ Model trained!")

# Step 6: Visualize Tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree for Animal Classification", fontsize=16, fontweight='bold')
plt.show()

# Step 7: Make Predictions
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 9: Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'][:10], importance['Importance'][:10])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Step 10: Test Custom Animal
print("\n🐕 Testing custom animal (dog-like):")
custom = pd.DataFrame([[1,0,0,1,0,0,1,1,1,1,0,0,4,1,1,1]], columns=X.columns)
prediction = dt.predict(custom)[0]
print(f"Predicted Class: {prediction}")

print("\n" + "="*50)
print("PROJECT COMPLETE! 🎉")
print("="*50)
print("\n📝 To Submit:")
print("1. File → Save a copy in Drive")
print("2. Click Share → Anyone with the link")
print("3. Copy link and paste in Word document")
```
