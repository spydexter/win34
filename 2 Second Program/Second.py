import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score

# Load the dataset
a = pd.read_csv("seattle-weather.csv")

# Drop the 'date' column
a = a.drop(['date'], axis=1)

# Encode the 'weather' column using LabelEncoder
l = LabelEncoder()
a["w"] = l.fit_transform(a["weather"])

# Drop the original 'weather' column
a = a.drop(["weather"], axis=1)

# Split data into features (x) and target (y)
x = a[["precipitation", "temp_max", "temp_min", "wind"]]
y = a["w"]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)

# Train a Linear Regression model
m = LinearRegression()
m.fit(x_train, y_train)

# Make a single prediction
print("Example prediction:", m.predict([[8.6, 4.4, 1.7, 1.3]]).round(0))

# Make predictions on the test set
y_predict = m.predict(x_test).round(0)

# Calculate F1 score
f1 = f1_score(y_test, y_predict, average='micro')
print("F1 score:", f1)

# Plot predictions
i = np.array(range(50))
plt.scatter(i, y_predict[0:50], label='Predicted')
plt.scatter(i, y_test.iloc[0:50], label='Actual')
plt.legend()
plt.show()
