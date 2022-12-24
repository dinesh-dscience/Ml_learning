import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

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

# Data split
attributes_train, cls_train, attributes_test, cls_test = sklearn.model_selection.train_test_split(attributes, cls, test_size=0.1)
