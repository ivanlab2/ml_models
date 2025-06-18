import numpy as np
from utils.preparation import standartize_data
from utils.metrics import euclide, max_n_em

class EM_clusterer():
    clusts=[]
    f=0#Минимальная схожесть разбиений
    mu_s=np.array([])#Средние кластеров
    matrix_s=np.array([])#Матрицы ковариаций
    w=np.array([])#Вектор весов
    g=np.array([])
    d=np.array([])
    def ro(self, x, m):#Расчёт расстояния от точки до центра кластера
        return np.sum(np.power(x-self.mu_s[m],2)*np.square(self.matrix_s[m]))
    
    def probability(self,x,m):#Расчёт вероятности принадлежности точки к кластеру
        return np.power(2*np.pi, -x.shape[0]/2)*1/np.linalg.det(self.matrix_s[m])*np.exp(-(1/2)*self.ro(x,m))
    
    def init_parameters(self, n_clusters,size):
        self.mu_s=np.random.normal(0,1,size=(n_clusters,size[1]))/5#Инициализация средних
        self.matrix_s=[np.diag(np.ones(size[1])) for i in range(n_clusters)]#Инициализация матрицы ковариаций
        self.g=np.zeros((size[0],n_clusters))#Инициализация матрицы вероятностей
        self.clusts=np.ones(size[0])#Инициализация вектора принадлежности кластерам
        self.w=np.ones(n_clusters)/n_clusters
        
    def fit(self, X, n_clusters, min_match=0.95, max_iters=100):
        self.d=standartize_data(X)
        self.init_parameters(n_clusters, self.d.shape)
        for step in range(max_iters):
        #E-шаг
            for j in range(self.d.shape[0]):
                g_j=np.array([])
                for i in range(n_clusters):
                    g_j=np.append(g_j, self.w[i]*self.probability(self.d[j,:],i))
                self.g[j,:]=g_j/np.sum(g_j)
            groups_new=np.argmax(self.g,axis=1)
            self.f=np.sum(groups_new==self.clusts)/self.clusts.shape[0]
            self.clusts=groups_new
        #M-шаг
            self.w=np.mean(self.g, axis=0)#Обновление весов
            for i in range(n_clusters):
                for j in range(self.d.shape[1]):
                    self.mu_s[i,j]=np.mean(self.g[:,i]*self.d[:,j])/self.w[i] #Обновление средних
            for i in range(n_clusters):
                for j in range(self.d.shape[1]):
                    self.matrix_s[i][j,j]=np.sqrt(np.mean(np.square(self.d[:,j]-self.mu_s[i,j])*self.g[:,i])/self.w[i])#Обновление матрицы ковариаций
            if self.f>min_match:
                break

    def predict_probabilities(self):#Возвращение вероятностей принадлежности точек к кластерам
        return self.g
    
    def predict_strict(self):#Жётский вариант кластеризации
        return self.clusts
    
    def average_between_clusters(self):#Расчёт среднего межкластерного расстояния
        cl_indexes=[]
        dists=np.array([])
        for cl in np.unique(self.clusts):
            cl_indexes.append(np.where(self.clusts==cl))
        for i in range(len(cl_indexes)):
            for j in range(i+1, len(cl_indexes)):
                dists=np.append(dists,max_n_em(cl_indexes,self.d,i,j)) 
        return np.mean(dists)
    def average_inside_clusters(self):#Расчёт среднего внутрикластерного расстояния
        cl_indexes=[]
        dists=np.array([])
        for cl in np.unique(self.clusts):
            cl_indexes.append(np.where(self.clusts==cl)[0])
        for cl in cl_indexes:
            ds=np.array([])
            for i in range(cl.shape[0]):
                for j in range(i+1, cl.shape[0]):
                    ds=np.append(ds,euclide(self.d[i],self.d[j]))
            dists=np.append(dists, np.mean(ds))
        return dists