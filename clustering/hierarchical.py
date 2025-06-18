import numpy as np
import matplotlib.pyplot as plt
import copy
from utils.preparation import standartize_data_to_matrix
from utils.metrics import euclide



def max_n(A,d, i_1,i_2):#Метод дальнего соседа
    m=-1
    for i in A[i_1]:
        for j in A[i_2]:
            t=euclide(d[i],d[j])
            if m<t:#Если нашелся сосед дальше, меняем максимальное расстояние
                m=t
    return m

def make_matrix(data, clusters):#Создание матрицы расстояний
    A=np.ones([len(clusters),len(clusters)])*100
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
                A[i,j]=max_n(clusters,data,i,j) 
    return A

def remake_matrix(m,i,j, clusters,data):#Пересчёт матрицы расстояний при измененении составов кластеров
    m=np.delete(m,j,0)#Удаление необходимых строки и столбца
    m=np.delete(m,j,1)
    for c in range(i):#Перерасчёт расстояний 
        m[c,i]=max_n(clusters,data,c,i)
    for c in range(i+1,m.shape[0]):
        m[i,c]=max_n(clusters,data,i,c)
    return m

class Hierarchical_clusterer():
    learning_history=[] #Объединяемые на каждой итерации кластеры
    clusts=[] #Состав кластеров на каждой итераыии
    average_between_clusters=[] #Среднее между кластерами
    dmatrix=[]#Матрица расстояний
    def fit(self,data): 
        e=[]
        d=standartize_data_to_matrix(data) #Стандартизация данных
        cl=np.arange(d.shape[0]).reshape((-1,1)).tolist() #Инициализация изначальных кластеров
        m=make_matrix(d, cl)#Создание матрицы расстояний
        self.dmatrix=m
        self.clusts=[]
        self.clusts.append(copy.deepcopy(cl))
        self.average_between_clusters.insert(0,np.mean(m, where=m !=100))
        for i in range(d.shape[0]-1):
            similar_clusters=[np.argmin(m)//m.shape[0],np.argmin(m)%m.shape[0]]#Запоминание координат ближайших друг к другу кластеров
            e.append([similar_clusters,np.min(m)])
            cl[similar_clusters[0]].extend(cl[similar_clusters[1]])#Обновление состава кластеров
            cl.pop(similar_clusters[1])
            self.average_between_clusters.insert(0,np.mean(m, where=m !=100))#Пересчёт среднего межкластерного расстояния
            if m.shape[0]!=1:
                m=remake_matrix(m,similar_clusters[0],similar_clusters[1],cl,d)#Пересчёт матрицы расстояний
            self.clusts.append(copy.deepcopy(cl))
        self.learning_history=e
        
    def draw_dendrogram(self):#Рисование дендрограммы
        dic={}#Словарь с координатами кластеров на плоскости
        clusts=self.clusts[-1][0]
        for i in range(len(clusts)):#Заполнение словаря
            dic[clusts[i]]=[i,0]
        fig=plt.figure(figsize = (20, 5))
        plt.xticks([i for i in range(len(clusts))],clusts, rotation=90) #Установка изначальных позиций кластеров
        max_dist=-1
        it=-1
        for i in range(len(self.clusts[0])-1):
            #Рисование вертикальных линий на объединяемых на каждом из шагов кластеров, высота - расстояние между кластерами
            plt.vlines(x = dic.get(self.learning_history[i][0][0])[0], ymax=self.learning_history[i][1], ymin=dic.get(self.learning_history[i][0][0])[1], color = 'b')
            plt.vlines(x = dic.get(self.learning_history[i][0][1])[0], ymax=self.learning_history[i][1],ymin=dic.get(self.learning_history[i][0][1])[1], color = 'b')
            if max_dist<max(self.learning_history[i][1]-dic.get(self.learning_history[i][0][0])[1],self.learning_history[i][1]-dic.get(self.learning_history[i][0][1])[1]):
                max_dist=max(self.learning_history[i][1]-dic.get(self.learning_history[i][0][0])[1],self.learning_history[i][1]-dic.get(self.learning_history[i][0][1])[1])
                it=i
            
            #Рисование горизонтальной линии между объединяемыми кластерами
            plt.hlines(xmin=dic.get(self.learning_history[i][0][0])[0], xmax=dic.get(self.learning_history[i][0][1])[0], y=self.learning_history[i][1])
            #Обновление координат кластеров
            dic.get(self.learning_history[i][0][0])[0], dic.get(self.learning_history[i][0][0])[1]=(dic.get(self.learning_history[i][0][0])[0]+dic.get(self.learning_history[i][0][1])[0])/2, self.learning_history[i][1]
            dic.pop(self.learning_history[i][0][1])
            #Обновление словаря
            for key in sorted(dic.keys()):
                if key>self.learning_history[i][0][1]:
                    dic[key-1] = dic.pop(key)
        plt.title(f'Максимальный прирост расстояния произошел при {len(self.clusts[0])-it+1} кластерах, его значение: {max_dist}')
        plt.show()
    def predict(self,k): #Предсказание для k кластеров
        clusters=self.clusts[-k]#Нужный состав кластеров из истории
        pr=np.arange(len(self.clusts[-1][0])).tolist()
        for i,obj in enumerate(clusters):#Вывод результатов
            for j in obj:
                pr[j]=i
        return pr
    def show_average_between_clusters(self, k):
        return self.average_between_clusters[k-1]
    def show_average_inside_clusters(self,k):#Расчёт среднего межкластерного расстояния
        clusters=self.clusts[-k]
        dim=[]
        for obj in clusters:
            mn=np.array([])
            clust=sorted(obj)
            if len(clust)!=1:
                for i in range(len(clust)):
                    for j in range(i+1, len(clust)):
                        mn=np.append(mn, self.dmatrix[i,j])
                dim.append(np.mean(mn))
            else: dim.append(0)
        return dim