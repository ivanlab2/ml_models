import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from utils.metrics import accuracy

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

class GradientBoostingClassifier():
    def __init__(self, n_trees, max_depth):
        self.n_trees=n_trees
        self.max_depth=max_depth

    def fit(self, X_train, y_train, return_fit_plot=False):
        predictions=np.zeros_like(y_train).astype(np.float32)#Начальное приближение
        self.trees=[]#Список из кортежей базовых алгоритмов и оптимальных размеров шага
        if return_fit_plot==True:#Визуализация процесса обучения
            accuracies=[]
        for _ in range(self.n_trees):
            antigradient=exp_loss_antigradient(y_train, predictions)#Вычисление антиградиента ошибок на векторе значений
            tree=DecisionTreeRegressor(criterion='squared_error', max_depth=self.max_depth)
            tree.fit(X_train, antigradient)#Обучение базового алгоритма на остатках
            prediction_new=tree.predict(X_train)#Предикт по остаткам
            gamma=grad_descent(y_train, prediction_new, predictions)#Поиск оптимального размера шага
            self.trees.append((tree, gamma))#Добавление дерева и шага в список
            predictions+=gamma*prediction_new#Обновление вектора значений
            if return_fit_plot==True:
               accuracies.append(accuracy(self.predict(X_train), y_train))
        if return_fit_plot==True:#Рисование графика процесса обучения
            plt.plot(accuracies)
            plt.title('Ход обучения')
            plt.xlabel('Число деревьев')
            plt.ylabel('Точность алгоритма')
            plt.grid()
            plt.show()

    def predict(self, X_test, threshold=0.5):
        logits=np.zeros(X_test.shape[0])#Инициализация векторов логитов
        for tree, gamma in self.trees:#Итеративное формирование вектора логитов
            logits+=tree.predict(X_test)*gamma
        probs=sigmoid(logits)#Перевод логитов в вероятности
        probs[probs>threshold]=1#Перевод вероятностей в классы
        probs[probs<=threshold]=-1
        return probs