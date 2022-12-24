import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


# Reading data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

attributes = np.array(data.drop(columns=[predict]))
label = np.array(data[predict])

attributes_train, attributes_test, label_train, label_test = sklearn.model_selection.train_test_split(attributes, label, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(attributes_train, label_train)
accuracy = linear.score(attributes_test, label_test)

print("Accuracy: %.2f%%" % (100 * accuracy))

predictions = linear.predict(attributes_test)

for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}, | Data: {attributes_test[i]} |  Actual Value:  {label_test[i]}")
