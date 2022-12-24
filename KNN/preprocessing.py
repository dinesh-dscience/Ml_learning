import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

# Read Data
data = pd.read_csv("car.data")

# Process irregular data into integer data
X = preprocessing.LabelEncoder()
buying = X.fit_transform(list(data["buying"]))
maint = X.fit_transform(list(data["maint"]))
doors = X.fit_transform(list(data["doors"]))
persons = X.fit_transform(list(data["persons"]))
lug_boot = X.fit_transform(list(data["lug_boot"]))
safety = X.fit_transform(list(data["safety"]))
cls = X.fit_transform(list(data["class"]))

predict = "class"
cls = list(cls)
attributes = list(zip(buying, maint, doors, persons, lug_boot, safety))
names = ["unacc", "acc", "good", "vgood"]

# Data split
attributes_train, attributes_test, cls_train, cls_test = sklearn.model_selection.train_test_split(attributes, cls, test_size=0.1)

model = KNeighborsClassifier(9)
model.fit(attributes_train, cls_train)

accuracy = model.score(attributes_test, cls_test)
print("Accuracy: %.2f%%" % (100 * accuracy))

# Predict
attributes_pred = model.predict(attributes_test)

for x in range(len(attributes_pred)):
    print('''
    Predicted Cls : {} \t | \t Data : {} \t | Actual:  \t {} |
    '''.format(names[attributes_pred[x]], attributes_test[x], names[cls_test[x]]))
