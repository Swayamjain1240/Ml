import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report

df = pd.read_csv("online_shooping.csv")

Pf = pd.DataFrame(df)

X = Pf.get_dummies(df)
Y = df["Revenue"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

pred = model.predict(x_test)

acc = accuracy_score(y_test, pred)
cr = classification_report(y_test, pred)

print(acc)
print(cr)