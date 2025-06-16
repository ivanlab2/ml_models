from sklearn.model_selection import KFold
import numpy as np

def find_most_frequent(preds):#Определение предсказания методом голосования
    for i in range(len(preds)):
        a,b=np.unique(preds,return_counts=True)
        return a[np.argmax(b)]
    
def accuracy(y_pred,y_test):#Точность
    return np.sum(y_pred==y_test)/len(y_test)

def cross_validation(randomforest, X,y,n_splits):#Кросс-валидация
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs=[]
    for trains,tests in kf.split(X):#n разбиений выборки на test и train
        randomforest.fit(X[trains,:],y[trains])
        accs.append(accuracy(randomforest.predict(X[tests,:]),y[tests]))#Оценивание на каждом разбиении
    return np.mean(accs)#Выдача среднего accuracy по каждому разбиению

def exp_loss_antigradient(y_real, y_pred): #Подсчёт антиградиента экспоненциальной функцией потерь
    return y_real*np.exp(-1*y_real*y_pred)

def sigmoid(x): #Сигмоида для перевода логитов в вероятности классов
    return 1/(1+np.exp(-x))

def grad_descent(y_real, preds, old_preds, start_value=0.5, step_size=0.005, n_iters=50, eps=0.001): #Градиентный спуск для поиска оптимального градиентного шага
    gamma=start_value#Начальное значение коэффициента
    for i in range(n_iters): #Градиентный спуск
        gradient=-np.sum(preds*y_real*np.exp(-y_real*(old_preds+gamma*preds))) #Расчёт градиента по гамме
        step=step_size*gradient #Шаг по градиенту
        gamma-=step
        if np.abs(step)<=eps: #Проверка на сходимость значения
            break
        if gamma<=0:
            print(f"Ошибка при минимизации: коэффициент шага оказался равным: ", {gamma})
            gamma=0
    return gamma

