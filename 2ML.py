import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data= {
    "word_count": [50, 120, 30, 90, 60, 150, 40, 200, 70, 110],
    "num_links": [0, 3, 0, 2, 1, 4, 0, 5, 1, 4],
    "capital_words": [1, 8, 0, 5, 2, 10, 0, 12, 1, 9],
    "spam_words": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
};

df=pd.DataFrame(data)

X = df[["word_count", "num_links", "capital_words", "spam_words"]]
Y = df[["label"]]

Xtrain, Xtest , Ytrain, Ytest = train_test_split(X,Y,test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(Xtrain,Ytrain)

yPred = model.predict(Xtest)

acc = accuracy_score(Ytest, yPred)
cm = confusion_matrix(Ytest, yPred)
report = classification_report(Ytest, yPred)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

new_email = pd.DataFrame({
    "word_count": [110],
    "num_links": [4],
    "capital_words": [10],
    "spam_words": [1]
})

prediction = model.predict(new_email)[0]

if prediction == 1 :
    print("New Email is Spam")
else:
    print("email is not Spam")