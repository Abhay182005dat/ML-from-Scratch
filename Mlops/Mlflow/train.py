import mlflow
import mlflow.sklearn 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import  matplotlib.pyplot as plt
import seaborn as sns 

# load the iris dataset 
iris = load_iris()
X = iris.data
y = iris.target 

# split the dataset into train and testing data sets
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.2 , random_state=42)

# define the parameters for the random forest model 
max_depth = 15 
n_estimators = 10

# Apply mlflow
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators , max_depth=max_depth)

    rf.fit(X_train , y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    print('Accuracy :' , accuracy)