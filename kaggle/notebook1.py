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
