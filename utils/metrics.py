from sklearn.model_selection import KFold
import numpy as np

    
def accuracy(y_pred,y_test):#Точность
    return np.sum(y_pred==y_test)/len(y_test)
'''
def cross_validation(randomforest, X,y,n_splits):#Кросс-валидация
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs=[]
    for trains,tests in kf.split(X):#n разбиений выборки на test и train
        randomforest.fit(X[trains,:],y[trains])
        accs.append(accuracy(randomforest.predict(X[tests,:]),y[tests]))#Оценивание на каждом разбиении
    return np.mean(accs)#Выдача среднего accuracy по каждому разбиению
'''
def create_model(model_class, **init_kwargs):
    def builder():
        return model_class(**init_kwargs)
    return builder


def cross_validation(model_builder, X, y, n_splits=5, fit_params=None, fit_predict=False):
    """
    model_class: класс модели или функция, возвращающая объект модели
    fit_params: словарь с параметрами для метода fit
    """
    if fit_params is None:
        fit_params = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kf.split(X):
        model = model_builder()  # создаём новую модель на каждую итерацию
        if fit_predict==True:
            y_pred=model.fit_predict(X[train_idx], y[train_idx],X[test_idx],**fit_params)
        else:
            model.fit(X[train_idx], y[train_idx], **fit_params)
            y_pred = model.predict(X[test_idx])
        accs.append(accuracy(y_pred, y[test_idx]))

    return np.mean(accs)


