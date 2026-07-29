import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = np.array([
    [17.99, 10.38, 122.8, 1001],
    [20.57, 17.77, 132.9, 1326],
    [19.69, 21.25, 130.0, 1203],
    [11.42, 20.38, 77.58, 386.1],
    [13.54, 14.36, 87.46, 566.3],
    [15.78, 17.89, 103.6, 781],
    [12.45, 15.70, 82.57, 477.1],
    [14.68, 20.13, 94.74, 684.5],
    [18.65, 17.60, 123.7, 1070],
    [11.90, 14.65, 78.83, 432.8],
    [16.02, 23.24, 102.7, 797.8],
    [12.36, 21.80, 79.01, 466.1]
])

Y = np.array([
    1,  # Malignant
    1,  # Malignant
    1,  # Malignant
    0,  # Benign
    0,  # Benign
    1,  # Malignant
    0,  # Benign
    0,  # Benign
    1,  # Malignant
    0,  # Benign
    1,  # Malignant
    0   # Benign
])

x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=42, test_size=0.2,stratify=Y)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = MLPClassifier(hidden_layer_sizes=(8,4), activation="relu", solver="adam", max_iter=2000, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(acc)
cm =confusion_matrix(y_test,y_pred)
print(cm)
cr = classification_report(y_test,y_pred,target_names=["Benign","Malignant"])
print(cr)

