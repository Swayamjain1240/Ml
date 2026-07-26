import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = {
    "age": [
        22, 25, 35, 40, 30,
        28, 45, 32, 26, 38,
        24, 50, 29, 41, 34
    ],

    "annual_income": [
        25000, 50000, 60000, 30000, 70000,
        55000, 35000, 65000, 45000, 80000,
        28000, 90000, 52000, 40000, 75000
    ],

    "spending_score": [
        30, 80, 50, 20, 75,
        85, 25, 70, 60, 90,
        35, 95, 78, 40, 82
    ],

    "purchases": [
        3, 15, 8, 2, 12,
        14, 4, 11, 9, 18,
        5, 20, 13, 6, 16
    ],

    "segment": [
        "Low", "High", "Medium", "Low", "High",
        "High", "Low", "High", "Medium", "High",
        "Low", "High", "High", "Medium", "High"
    ]
}

df = pd.DataFrame(data)
print("dataset : " , df)

X = df [["age","annual_income","spending_score","purchases"]]
Y = df["segment"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42,stratify=Y)

model = RandomForestClassifier(n_estimators=100,random_state=42) #estimators -> creating 100 decision tree
model.fit(x_train, y_train)

y_pred = model.predict(x_train)

#evolution 

acc = accuracy_score(y_train, y_pred)
cr = classification_report(y_train, y_pred, zero_division=0)
cm = (y_test, y_pred)

new_customer = pd.DataFrame({
    "age": [28],
    "annual_income": [55000],
    "spending_score": [85],
    "purchases": [14]
})

prediction = model.predict(new_customer)

print("\nNew Customer Prediction:")
print(prediction[0])