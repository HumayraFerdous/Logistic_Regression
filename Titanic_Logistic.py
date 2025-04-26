import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt



data = pd.read_csv("Titanic-Dataset.csv")
#print(data.head())
#print(data.isnull().sum())
#print(data.dtypes)
data['Age']= data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#print(data.isnull().sum())

X = data.drop('Survived',axis=1)
y = data['Survived']

numeric_features = ['Age','Fare','SibSp','Parch']
categorical_features = ['Pclass','Sex','Embarked']

numeric_trans = StandardScaler()
categorical_trans = OneHotEncoder(drop='first')

preprocess = ColumnTransformer(
    transformers=[('num',numeric_trans,numeric_features),('cat',categorical_trans,categorical_features)]
)

pipeline = Pipeline(steps = [('preprocessor',preprocess),
                             ('classifier',LogisticRegression(max_iter=1000))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_pred_proba = pipeline.predict_proba(X_test)[:,1]
fpr,tpr, thresholds = roc_curve(y_test,y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()
