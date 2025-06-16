import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier as EtalonClassifier
from ensembles import RandomForest, GradientBoostingClassifier
from utils import cross_validation

data=pd.read_csv('../data/breast-cancer.csv', sep=',')
data[(data[['diagnosis']]=='M')]=1
data[(data[['diagnosis']]=='B')]=-1
data=data.drop(['id'], axis=1)
y=data[['diagnosis']].to_numpy().reshape(-1).astype(np.int8)
data=data.drop(['diagnosis'], axis=1)
data=data.drop(['radius_mean', 'area_mean', 'radius_mean', 'radius_worst', 'area_worst', 'perimeter_worst'], axis=1)
data=data.drop(['concavity_mean', 'concave points_mean', 'texture_worst'], axis=1)
data=data.drop(['area_se', 'perimeter_se', 'compactness_worst','concave points_worst'], axis=1)
data=data.drop(['smoothness_worst','concavity_worst'], axis=1)
X=data.to_numpy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=24)

rf=RandomForest(n_trees=130, max_depth=10, n_features=4,min_samples_leaf=2,min_samples_split=5, eps_1=0.55, eps_2=0.55)
start = time.time()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
end = time.time()
print("Время обучения и предсказания собственной реализации случайного леса:",
      round((end-start) * 10**3,3), "ms")

clf=RandomForestClassifier(max_depth=10, criterion='entropy', n_estimators=130, min_samples_leaf=2,min_samples_split=5)
start = time.time()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
end = time.time()
print("Время обучения и предсказания эталонной реализации случайного леса:",
      round((end-start) * 10**3,3), "ms")

gb=GradientBoostingClassifier(n_trees=300, max_depth=1)
start = time.time()
gb.fit(X_train,y_train)
y_pred=gb.predict(X_test)
end = time.time()
print("Время обучения и предсказания собственной реализации градиентного бустинга:",
      round((end-start) * 10**3,3), "ms")

clf_gb = EtalonClassifier(n_estimators=300, loss='exponential',criterion='squared_error',
    max_depth=1, random_state=0)
start = time.time()
clf_gb.fit(X_train,y_train)
y_pred=clf_gb.predict(X_test)
end = time.time()
print("Время обучения и предсказания эталонной реализации градиентного бустинга:",
      round((end-start) * 10**3,3), "ms")

print("Точность написанного вручную случайного леса с использованием кросс-валидации:", np.round(cross_validation(rf,X,y,5),4))
print("Точность эталонной реализации случайного леса с использованием кросс-валидации:", np.round(cross_validation(clf,X,y,5),4))
print("Точность написанного вручную градиентного бустинга с использованием кросс-валидации:", np.round(cross_validation(gb,X,y,5),4))
print("Точность эталонной реализации градиентного бустинга с использованием кросс-валидации:", np.round(cross_validation(clf_gb,X,y,5),4))