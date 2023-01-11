import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import datasets

iris = datasets.load_dataset()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Target'] = iris.target


X_train, X_test, y_train, y_test = train_test_split(data.drop('Target', axis=1), data['Target'], test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
joblib.dump(lr, 'lr_mode.pkl')
