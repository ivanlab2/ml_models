import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def standartize_data(data): #Стандартизация данных
    d=(data-data.mean())/data.std()
    return d.to_numpy()

def euclide(x_t,x):
    return np.sum(np.power((x_t[:,None]-x),2),axis=-1)

def gauss_kernel(dists, k):#Расчёт Гауссова ядра
    return 1 / np.sqrt(2*np.pi)*np.exp(-((dists[:,:k]/dists[:,k].reshape(dists.shape[0],-1)**2)/2))
    
def abs_dist(x_t,x): #L1-метрика
    return np.sum(np.abs(x_t[:,None]-x),axis=-1)

class KNN():
    def fit_predict(self, x_train, y_train, x_test, k, method='euclide'):
        if method=='euclide':
            dists=euclide(x_test, x_train)#Расстояния от вводимых точек до обучающих
        else:
            dists=abs_dist(x_test, x_train)
        s_dists=np.sort(dists)[:,:k+1]#Отсортированные k ближайших к вводимым обучающих точек 
        s_idx=np.argsort(dists)[:,:k]#Индексы отсортированных точек
        weights=gauss_kernel(dists,k)#Веса объектов
        marks=[]#Итоговые метки классов
        for i in range(s_idx.shape[0]): #Итерируемся по каждому тестовому объекту
            u=y_train[s_idx[i,:k]] #Метки классов k ближайших точек
            data=defaultdict() 
            for j in range(k):#Объявление и заполнение словаря метка класса:суммарный вес
                data[u[j]]=0
            for j in range(k):
                data[u[j]]+=weights[i,j] #Добавление сумм весов в массив
            marks.append(max(data, key=data.get))#Итоговый ответ     
        return marks

    def leave_one_out(self,x,y, k, method='euclide'):
        marks=[]
        for i in range(x.shape[0]):#Подготовка данных и обучение
            x_test=np.array([x[i,:]])
            x_train=np.delete(x,i,axis=0)
            y_test=y[i]
            y_train=np.delete(y,i,axis=0)
            marks.append(self.fit_predict(x_train, y_train, x_test,k, method)[0])
        return np.sum(marks==y)/len(marks)#Возвращение точности классификации        

    def build_graph(self,x,y, method='euclide'):#Построение графика
        precisions=[]
        for i in range(1, 101):
            precisions.append(self.leave_one_out(x,y,i, method))
        plt.plot([i for i in range(1,101)],precisions)
        plt.xlabel('Число соседей')
        plt.ylabel('Точность классификации')
        plt.title('График эмпирического риска')
        plt.show()      