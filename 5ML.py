import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = {
    "age": [
        22, 35, 28, 45, 25,
        30, 40, 27, 50, 32,
        24, 38, 29, 48, 26,
        42, 31, 23, 36, 44
    ],

    "monthly_charges": [
        90, 50, 80, 40, 75,
        85, 45, 95, 35, 70,
        88, 55, 92, 42, 78,
        48, 82, 65, 60, 38
    ],

    "tenure_months": [
        2, 24, 4, 36, 5,
        3, 30, 2, 48, 12,
        4, 20, 3, 40, 6,
        28, 5, 18, 15, 45
    ],

    "support_tickets": [
        7, 1, 6, 0, 8,
        6, 1, 9, 0, 3,
        7, 2, 8, 1, 6,
        1, 7, 3, 4, 0
    ],

    "churn": [
        1, 0, 1, 0, 1,
        1, 0, 1, 0, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 0, 0
    ]
}

df = pd.DataFrame(data)


X = df[["age",
        "monthly_charges",
        "tenure_months",
        "support_tickets"]]

Y = df["churn"]


x_train, x_test, y_train , y_test = train_test_split(X,Y,test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)

y_pred = model.predict(x_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr =  classification_report(y_test,y_pred,zero_division=0)

new_customer = pd.DataFrame({
    "age": [25],
    "monthly_charges": [80],
    "tenure_months": [3],
    "support_tickets": [8]
})

new_customer_scaled = scaler.transform(
    new_customer
)

prediction = model.predict(
    new_customer_scaled
)



if prediction[0] == 1:
    print("\nPrediction: Customer will Churn")
else:
    print("\nPrediction: Customer will Stay")