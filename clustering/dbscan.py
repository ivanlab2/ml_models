import numpy as np
from utils.preparation import standartize_data
from utils.metrics import euclide, max_n_em


def make_matrix(d):#Создание матрицы расстояний
    A=np.diag(np.ones(d.shape[0])*np.inf)
    for i in range(d.shape[0]):
        for j in range(i+1, d.shape[0]):
                A[i,j]=euclide(d[i],d[j])
                A[j,i]=A[i,j]
    return A

def remake_matrix(A,cl):#Пересчёт матрицы расстояний при изменении кластерного состава
    for i in cl:
        A[cl,:],A[:,cl]=np.inf,np.inf
    return A


class DBSCAN_clusterer():
    dic={}
    predictions=[]
    def init_dict(self,d):
        for i in range(d.shape[0]):
            self.dic[i]=[0,0]
            
    def fit_predict(self,data, eps, m):#Обучение и предсказание
        d=standartize_data(data)#Стандарта=изация данных
        self.init_dict(d)#Создание словаря "Объект:[кластер, тип точки]"
        U=np.arange(d.shape[0])#Создание множества U
        A=make_matrix(d)#Создание матрицы расстояний
        clusters_count=0
        while len(U)>0:#Пока U непустое
            Ux=np.where(A[U[0]]<eps)[0]#Поиск всех точек в eps-окрестности
            if len(Ux)>=m:#Если таких точек больше, чем m
                cl=np.array([U[0]])#Создание кластера
                self.dic.get(U[0])[0],self.dic.get(U[0])[1]=clusters_count,2#Обновление словаря
                while len(Ux)>0:#Бежим по всем граничным точкам
                    el=Ux[0]
                    cl=np.append(cl,el)#Добавление точки в кластер
                    Ux=np.delete(Ux,0)#Её удаление из граничных
                    Ut=np.where(A[el]<eps)[0]#РАсчёт eps-окрестности
                    if len(Ut)>=m:#Если точек больше, чем m, то это корневая точка
                        self.dic.get(el)[0],self.dic.get(el)[1]=clusters_count,2
                        Ux=np.unique(np.append(Ux,Ut))#Пополняем массив граничных точек
                        indices=np.argwhere(np.isin(Ux,cl))
                        Ux=np.delete(Ux,indices)#Избавление от дубликатов
                    else:#Если нет, то оставляем граничной
                        self.dic.get(el)[0],self.dic.get(el)[1]=clusters_count,1
                U=np.delete(U,np.argwhere(np.isin(U,cl)))#Удаление из множества непомеченных точек всех из кластера
                A=remake_matrix(A,cl.astype(int))#Пересчитываем матрицу расстояний, удаляя все точки из кластера
                clusters_count+=1
            else:#Если это шумовая точка
                self.dic.get(U[0])[0]=-1
                U=np.delete(U,0)#Удаление из множества непомеченных точек
        self.predictions=[self.dic.get(i)[0] for i in range(len(self.dic.keys()))]
        return self.predictions
    def show_object_types(self):#Возвращаем типа точек
        tps=[]
        for i in range(len(self.dic.keys())):
            if self.dic.get(i)[1]==0:
                tps.append('Outlier')
            elif self.dic.get(i)[1]==1:
                tps.append('Non-core')
            else: tps.append('Core')
        return tps
    def average_between_clusters(self,data):#Расчёт среднего межкластерного расстояния
        cl_indexes=[]
        dists=np.array([])
        for cl in np.unique(np.array(self.predictions)):
            cl_indexes.append(np.where(np.array(self.predictions)==cl))
        for i in range(len(cl_indexes)):
            for j in range(i+1, len(cl_indexes)):
                dists=np.append(dists,max_n_em(cl_indexes,standartize_data(data),i,j)) 
        return np.mean(dists)
    def average_inside_clusters(self,data):#Расчёт среднего внутрикластерного расстояния
        cl_indexes=[]
        dists=np.array([])
        for cl in np.unique(np.array(self.predictions)):
            cl_indexes.append(np.where(np.array(self.predictions)==cl)[0])
        for cl in cl_indexes:
            ds=np.array([])
            for i in range(cl.shape[0]):
                for j in range(i+1, cl.shape[0]):
                    ds=np.append(ds,euclide(standartize_data(data)[i],standartize_data(data)[j]))
            dists=np.append(dists, np.mean(ds))
        return dists[1:]