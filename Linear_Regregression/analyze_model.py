import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

# Open Model
pickle_student_model = open("student_grade_predictions", "rb")
linear = pickle.load(pickle_student_model)

# Reading data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Define styles and plots
style.use("ggplot")

# Define predictors to find correlation
predictor = "G1"
prediction = "G3"

plt.scatter(data[predictor], data[prediction])
plt.xlabel("Grade 1")
plt.ylabel("Final Grade")
plt.title("Grade Prediction")
plt.show()
