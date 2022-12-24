import pickle

pickle_student_model = open("student_grade_predictions", "rb")
linear = pickle.load(pickle_student_model)

print("\nFinal Accuracy: %.2f%%" % (100 * ACCURACY))

predictions = linear.predict(attributes_test)

# Printing predictions alongside the actual data
for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}, | Data: {attributes_test[i]} |  Actual Value:  {label_test[i]}")