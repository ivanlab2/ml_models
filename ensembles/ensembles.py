import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt
from utils import *

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
    
class RandomForest():
    def __init__(self, n_trees, max_depth,  n_features,  min_samples_leaf=1, min_samples_split=2, criterion='entropy', eps_1=0.95, eps_2=0.9):
        self.n_trees=n_trees
        self.n_features=n_features
        self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf
        self.min_samples_split=min_samples_split
        self.criterion=criterion
        self.eps_1=eps_1 #Критерий включения дерева по точности на обучающей выборке
        self.eps_2=eps_2 #Критерий включения дерева по точности на отложенной выборке
    def fit(self,X,y):
        self.trees=[]
        while len(self.trees)<self.n_trees:
            tree=DecisionTreeClassifier(max_features=self.n_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,criterion=self.criterion)
            t=np.random.choice(np.arange(X.shape[0]),size=(np.round(X.shape[0]*0.6).astype('int')), replace=True)
            tree.fit(X[t,:],y[t])
            if (accuracy(tree.predict(X[t,:]),y[t])>=self.eps_1) and (accuracy(tree.predict(X[~t,:]),y[~t])>=self.eps_2):
                self.trees.append(tree)
        
    def predict(self,X):
        preds=[]
        for tree in self.trees:#Предсказание по каждому дереву
            preds.append(tree.predict(X))
        y_pred=np.apply_along_axis(find_most_frequent, axis=0,arr=np.array(preds))#Определение итогового предсказания голосованием
        return y_pred