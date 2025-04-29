import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Trojan_Detection.csv")
#print(data.head())
#print(data.dtypes)
#print(data.shape)
#print("Missing values: ", data.isnull().sum())

data["Class"] = data["Class"].map({'Benign': 0, 'Trojan': 1})
data['Class'] = data['Class'].astype(int)
data.drop(['Unnamed: 0', 'Flow ID'],axis=1,inplace=True)
print(data.dtypes)

encoder = LabelEncoder()
data[' Source IP'] = encoder.fit_transform(data[' Source IP'])
print(data[' Source IP'].nunique())
data[' Destination IP'] = encoder.fit_transform(data[' Destination IP'])
print(data[' Destination IP'].nunique())

data[' Timestamp'] = pd.to_datetime(data[' Timestamp'], format='%d/%m/%Y %H:%M:%S')

data['year'] = data[' Timestamp'].dt.year
data['month'] = data[' Timestamp'].dt.month
data['day'] = data[' Timestamp'].dt.day
data['hour'] = data[' Timestamp'].dt.hour
data['minute'] = data[' Timestamp'].dt.minute
data['weekday'] = data[' Timestamp'].dt.weekday  # 0 = Monday, 6 = Sunday
data['dayofyear'] = data[' Timestamp'].dt.dayofyear

data.drop(' Timestamp',axis=1,inplace=True)

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(
    pd.crosstab(y_test, y_pred),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Benign', 'Trojan'],
    yticklabels=['Benign', 'Trojan']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()