# Decision Tree - Supervised Learning - iris flower - 'versicolor' 'virginica' [1 , 2]
# data : 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt


# 1. load the dataset
iris = load_iris(as_frame=True) # as_frame=True change the data to Pandas DataFrame

'''print(iris.data.shape)
print(iris.target_names)
print(iris.feature_names)
print(iris.data.head(5))'''

# 2. Data Preparation
X =  iris.data # Independent variables
y = iris.target # classes(labels)

# Binomial data
mask = (y==1)|(y==2)
X = X[mask]
y = y[mask]
print("Classes:", iris.target_names[np.unique(y)]) # name of the classes

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2 , random_state=42, stratify=y)
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# 3. Model creation
dtree = DecisionTreeClassifier(random_state=42)
dtree = dtree.fit(train_x, train_y)


# 4. Prediction
y_pred = dtree.predict(test_x)
print("y_predicted:", y_pred)

# 5. Evaluation

# Confusion Matrix
cm = metrics.confusion_matrix(test_y, y_pred) # test_y: actual, y_pred: predicted

Accuracy = metrics.accuracy_score(test_y, y_pred)
Precision = metrics.precision_score(test_y, y_pred)
Sensitivity_recall = metrics.recall_score(test_y, y_pred)
Specificity = metrics.recall_score(test_y, y_pred)
F1_score = metrics.f1_score(test_y, y_pred)


# 6. Print the result
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})
print(metrics.classification_report(test_y, y_pred))
print("Confusion Matrix:\n", cm)

# 7. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # figsize=(width, height)

# Plot 1: Confusion Matrix
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names[np.unique(y)]
)
cm_display.plot(ax=axes[0])
axes[0].set_title("Confusion Matrix - Versicolor vs Virginica")

# Plot 2: Decision Tree
plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=iris.target_names[np.unique(y)],
    filled=True,
    ax=axes[1]
)
axes[1].set_title("Decision Tree")

plt.tight_layout()
plt.show()
