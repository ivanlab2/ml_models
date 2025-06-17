import numpy as np

class GaussianNBC():
    def __init__(self):
        pass
    def fit(self, X_train,y_train):
        vals, counts=np.unique(y_train, return_counts=True) #Получение всех значений и частоты их попадания
        self.params={}#Словарь значение: его средние, стандартные отклонения и вероятности
        for i, value in enumerate(vals):#Каждое из значений добавляем в словарь
            self.params[value]=(np.mean(X_train[np.where(y_train==value)], axis=0), np.std(X_train[np.where(y_train==value)], axis=0), np.log(counts[i])/np.sum(counts))
    def predict(self, X_test):
        probabilities=[]#Массив с формулами по каждому из значений
        for target in self.params.keys():
            params=self.params[target]#Получаем параметры по каждому значению
            probability=params[2]-np.sum(((X_test-params[0])**2)/(2*params[1]**2)+np.log(params[1]),axis=1)#Расчёт по формуле
            probabilities.append(probability)#Добавление результатов
        return np.array(list(self.params.keys()))[np.argmax(np.array(probabilities), axis=0)]#Возвращается класс, соответствующий индексу аргмакса результата по формуле