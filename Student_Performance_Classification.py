import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve
data = {
    "Study_Hours": [2,3,4,5,1,2,3,4,5,6,1,2,3,4,5,6,7,3,4,5],
    "Sleep_Hours": [6,7,8,7,5,6,7,8,7,8,4,5,6,7,8,7,8,5,6,7],
    "Pass":[0,0,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,0,1,1]
}
df = pd.DataFrame(data)
X = df[["Study_Hours","Sleep_Hours"]]
y = df["Pass"]
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:,1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

fpr,tpr,thresholds = roc_curve(y_test,y_proba)
plt.plot(fpr,tpr,label=f"AUC = {roc_auc_score(y_test,y_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

new_threshold = 0.4
y_pred_adjusted = (y_proba>=new_threshold).astype(int)
print("Adjusted Threshold Report (0.3):\n",classification_report(y_test,y_pred_adjusted))