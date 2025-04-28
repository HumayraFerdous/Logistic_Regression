import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("winequality-red.csv")
#print(data.head())
#print(data.dtypes)
#print(data.shape)
#print(data.isnull().sum())

sns.countplot(x='quality',data=data)
plt.title('Wine Quality Distribution')
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

X = data.drop('quality',axis=1)
y = data['quality']

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver='lbfgs',max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

def quality_group(q):
    if q <= 5:
        return 0
    elif q == 6:
        return 1
    else:
        return 2
data['quality_group'] =data['quality'].apply(quality_group)
#print(data.head())

X2 = data.drop(['quality', 'quality_group'], axis=1)
y2 = data['quality_group']

# Train-Test Split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)


model = LogisticRegression(solver='lbfgs', max_iter=2000)
model.fit(X2_train, y2_train)

y_pred2 = model.predict(X2_test)

print("Accuracy after grouping:", accuracy_score(y2_test, y_pred2))
print("\nClassification Report after grouping:\n", classification_report(y2_test, y_pred2))

conf_mat = confusion_matrix(y2_test, y_pred2)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix (Grouped Classes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()