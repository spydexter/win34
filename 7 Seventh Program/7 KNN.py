from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

# Print the Iris data and labels
print("Iris Data:\n", iris_data)
print("Iris Labels:\n", iris_labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.30)

# Initialize and train the K-Nearest Neighbors classifier
clsify = KNeighborsClassifier(n_neighbors=5)
clsify.fit(x_train, y_train)

# Predict the labels for the test set
y_pred = clsify.predict(x_test)

# Print the confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
