import pickle
import sys
from time import sleep
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

ACCURACY = 0

# Reading data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

attributes = np.array(data.drop(columns=[predict]))
label = np.array(data[predict])

for i in range(100):
    attributes_train, attributes_test, label_train, label_test = sklearn.model_selection.train_test_split(attributes, label, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(attributes_train, label_train)
    accuracy = linear.score(attributes_test, label_test)

    sys.stdout.write("\rAccuracy: %.2f%%" % (100 * accuracy))
    sys.stdout.flush()
    sleep(0.09)

    if accuracy > ACCURACY:
        ACCURACY = accuracy
        with open("student_grade_predictions", "wb") as f:
            pickle.dump(linear, f)

print("\nFinal Accuracy: %.2f%%" % (100 * ACCURACY))

predictions = linear.predict(attributes_test)

# Printing predictions alongside the actual data
for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}, | Data: {attributes_test[i]} |  Actual Value:  {label_test[i]}")