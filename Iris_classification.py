import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
data = pd.read_csv("Iris.csv")
#print(data.head())
#print(data.dtypes)
#print(data.shape)
data.drop("Id",axis =1,inplace=True)
plt.figure(figsize=(8,4))
print(data["Species"].value_counts())
sns.countplot(data["Species"])
plt.show()
x = data.drop("Species",axis=1)
plt.figure(figsize=(8,4))
sns.heatmap(x.corr(),annot=True,fmt=".0%")
plt.show()

sns.FacetGrid(data, hue="Species", height=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
plt.show()

sns.pairplot(data.iloc[:,:],hue="Species")
plt.show()

data = data[data['Species']!='Iris-setosa']
data.reset_index(drop=True,inplace=True)
data['Species'] = data['Species'].map({'Iris-versicolor': 0, 'Iris-virginica': 1})
X = data.drop("Species", axis=1)
y = data["Species"]

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled,y_train)
y_pred = log_reg.predict(X_test_scaled)


accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)

print("Accuracy: ",accuracy)
print("Classification Report: \n",report)

conf_matrix = pd.crosstab(
    pd.Series(y_test.values, name='Actual'),
    pd.Series(y_pred, name='Predicted')
)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix (Versicolor vs Virginica)')
plt.show()



