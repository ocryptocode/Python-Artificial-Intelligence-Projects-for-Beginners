import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('student-por.csv', sep=';')

# Generate binary label (pass/fail)
data['pass'] = (data['G1'] + data['G2'] + data['G3']) >= 35

# One-hot encoding of categorical columns
data = pd.get_dummies(data, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                                      'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                                      'nursery', 'higher', 'internet', 'romantic'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['pass'], axis=1), data['pass'], test_size=0.2, random_state=42)

# Fit a decision tree classifier
tree_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
tree_classifier.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree_classifier, feature_names=X_train.columns, class_names=["fail", "pass"], filled=True)
plt.show()

# Evaluate the model on the testing set
y_pred = tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Cross-validation for model evaluation
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(tree_classifier, data.drop(['pass'], axis=1), data['pass'], scoring=scoring, cv=5)
print("Cross-Validation Results:")
for metric in scoring:
    print(f"{metric.capitalize()}: {cv_results['test_'+metric].mean()} +/- {2 * cv_results['test_'+metric].std()}")
