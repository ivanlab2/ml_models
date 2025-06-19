import numpy as np
import pandas as pd
from utils.preparation import split_data
from utils.metrics import mse
import copy


def mse_crit(y, y_u):#Подсчёт MSE-критерия
    return np.min(np.sum(np.square(y.reshape(-1,1)-y_u),axis=1)/len(y_u))

def get_optimal_split_r(x,y,y_all):#Выбор оптимального разделения для регрессии
    max_entropy=-1
    value=0
    feat=0
    ent_start=mse_crit(y_all,y)
    for feature in range(x.shape[1]):
        nans_idx=np.where(pd.isnull(x[:,feature]))
        nans_x=x[nans_idx]
        nans_y=y[nans_idx]
        x=np.delete(x, nans_idx, axis=0)
        y=np.delete(y, nans_idx, axis=0)
        values=np.unique(x[:,feature])[1:]
        for v in values:#Реализация поиска оптимального разбиения по MSE-критерию
            _,_, y_r,y_l=split_data(x,y,feature,v)
            entropy=ent_start-mse_crit(y_all,y_r)*len(y_r)/len(y)-mse_crit(y_all,y_l)*len(y_l)/len(y)
            if entropy>max_entropy:
                feat=feature
                value=v
                max_entropy=entropy
        x=np.concatenate((x,nans_x), axis=0)
        y=np.concatenate((y,nans_y), axis=0)
    nans_idx=np.where(pd.isnull(x[:,feat]))
    nans_x=x[nans_idx]
    nans_y=y[nans_idx]
    x=np.delete(x, nans_idx, axis=0)
    y=np.delete(y, nans_idx, axis=0)
    X_r, X_l, y_r,y_l=split_data(x,y,feat,value)
    q=len(X_l)/(len(X_r)+len(X_l))
    rans=np.array(np.random.choice([0,1], len(nans_x), replace=True, p=[q,1-q]))
    X_l=np.concatenate((X_l,nans_x[np.where(rans==0)]),axis=0)
    y_l=np.concatenate((y_l,nans_y[np.where(rans==0)]),axis=0)
    X_r=np.concatenate((X_r,nans_x[np.where(rans==1)]), axis=0)
    y_r=np.concatenate((y_r,nans_y[np.where(rans==1)]),axis=0)
    return feat, value, (X_r, X_l, y_r,y_l)

def get_average_value(rules,level,key):#Получение среднего по листу значения
    vals=[]
    for spl in rules[level+1]:
        if spl[0]==key+'0' or spl[0]==key+'1':
            if spl[1]=='leaf':
                vals.append([spl[2],spl[3]])
            else:
                for v in get_average_value(rules,level+1,spl[0]):
                    vals.append(v)
    return vals
    
class DecisionTreeRegressor:#Дерево регрессии
    def __init__(self, depth):
        self.depth = depth
    rules=[]
    splits=[]
    def fit(self, x,y):
        splits=[[['0',x,y]]]
        rules=[]
        k=1
        while k<=self.depth and k==len(splits):#Пока дерево растёт вглубь и не достигнута максимальная глубина
            p=[]
            r=[]
            for split in splits[k-1]:
                feature, value, [X_r, X_l, y_r,y_l]= get_optimal_split_r(split[1],split[2], y)
                if len(y_r)==0 or len(y_l)==0:
                    r.append([split[0],'leaf',np.mean(split[2]),len(split[2])])#При попадании в лист считаем среднее значение таргетов элементов
                else:#При попадании в узел формируем новые ветки
                    p.append([split[0]+'0',X_l,y_l])
                    p.append([split[0]+'1',X_r,y_r])
                    r.append([split[0], 'node', feature, value])
            rules.append(r)
            k+=1
            if len(splits)<self.depth:
                splits.append(p)
        for i in range(len(splits[-1])):#По окончании цикла превращаем все разбиения на максимальной глубине в листья
            rules[-1][i]=[rules[-1][i][0],'leaf',np.mean(splits[-1][i][2]),len(splits[-1][i][2])]
        self.splits=splits
        self.rules=rules
        
    def predict(self,x):
            predict=[]
            for i in range(len(x)):
                xt=x[i,:]
                k_start='0'
                level=0
                flg=True
                while flg==True:
                    for m in self.rules[level]:
                        if m[0]==k_start:#При попаданиив лист присваеваем объекту среднее значение в листе
                            if m[1]=='leaf':
                                predict.append(m[2])
                                flg=False
                            elif pd.isnull(xt[m[2]])==True:#Если значение пустое, присваем средневзвешенное по листьям ниже узла
                                summa=np.sum(np.array([a[0]*a[1] for a in get_average_value(self.rules,level,k_start)]))
                                kol=np.sum(np.array([a[1] for a in get_average_value(self.rules,level,k_start)]))
                                predict.append(summa/kol)
                                flg=False    
                            else:#Иначе - идём дальше по дереву
                                try:
                                    if xt[m[2]]>=m[3]:
                                        k_start=k_start+'1'
                                    else:
                                        k_start=k_start+'0'
                                    level+=1 
                                except TypeError:
                                    if xt[m[2]]==m[3]:
                                        k_start=k_start+'1'
                                    else:
                                        k_start=k_start+'0'
                                    level+=1    
            return predict  
