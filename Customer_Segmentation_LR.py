import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

data = pd.read_csv('Customer_Segmentation.csv')
#print(data.head())
#print(data.shape)
#print(data.isnull().sum())
print(data.dtypes)
data['Ever_Married'] = data['Ever_Married'].fillna(data['Ever_Married'].mode()[0])
data['Graduated'] = data['Graduated'].fillna(data['Graduated'].mode()[0])
data['Profession'] = data['Profession'].fillna(data['Profession'].mode()[0])
data['Work_Experience'] = data['Work_Experience'].fillna(data['Work_Experience'].median())
data['Family_Size'] = data['Family_Size'].fillna(data['Family_Size'].median())
data['Var_1'] = data['Var_1'].fillna(data['Var_1'].mode()[0])
#(data.isnull().sum())
print(data['Segmentation'].value_counts())
print(data['Profession'].value_counts())
data.drop('ID',axis=1,inplace=True)
le = LabelEncoder()
scaler = MinMaxScaler()
cat_col = ['Gender', 'Ever_Married', 'Graduated', 'Profession','Spending_Score', 'Var_1']
num_col = ['Age','Work_Experience','Family_Size']
for i in cat_col:
    data[i] = le.fit_transform(data[i])
for i in num_col:
    data[[i]] = scaler.fit_transform(data[[i]])

data['Segmentation']= data['Segmentation'].map({'A':0,'B':1,'C':2,'D':3}).astype(int)
print(data.head())

X = data.drop('Segmentation',axis=1)
y = data['Segmentation']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LogisticRegression(solver='lbfgs',class_weight='balanced',max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['A', 'B','C','D'],
            yticklabels=['A','B','C','D'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


