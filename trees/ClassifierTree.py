import numpy as np
import pandas as pd
from utils.preparation import split_data
import copy

def m_entropy(x):#Функция подсчёта энтропии
    if x==0:
        return 0
    else:
        return -x*np.log2(x)


def get_optimal_split(x,y, crit='e'):#Получение оптимального разбиения
    max_entropy=-1
    value=0
    feat=0
    for feature in range(x.shape[1]):
        nans_idx=np.where(pd.isnull(x[:,feature]))#Временное удаление пропусков из выборки
        nans_x=x[nans_idx]
        nans_y=y[nans_idx]
        x=np.delete(x, nans_idx, axis=0)
        y=np.delete(y, nans_idx, axis=0)
        values=np.unique(x[:,feature])[1:]#Выделение всех уникальных значений признака
        if crit=='d':#Критерий Донского
            for v in values:#Для каждого разделения смотрим информативность
                X_r, X_l, y_r,y_l=split_data(x,y,feature,v)#Разделение данных
                entropy=0
                for i in range(len(X_l)):
                    entropy+=np.sum(y_l[i]!=y_r)#Подсчёт разных значений для каждого элемента
                if entropy>max_entropy:#Если информативность наибольшая, запоминаем
                    feat=feature
                    value=v
                    max_entropy=entropy
        else:#Энтропийный критерий
            for v in values:#Для каждого разделения смотрим информативность
                X_r, _, y_r,_ = split_data(x,y,feature,v)#Разделение данных
                entropy=0
                for i in np.unique(y):#Подсчёт энтропии по формуле
                    l=x.shape[0]
                    p=X_r.shape[0]
                    Pc=y[np.where(y ==i)].shape[0]
                    pc=y_r[np.where(y_r ==i)].shape[0]
                    entropy+=m_entropy(Pc/l)-p/l*m_entropy(pc/p)-(l-p)/l*m_entropy((Pc-pc)/(l-p))
                if entropy>max_entropy:#Если информативность наибольшая, запоминаем
                    feat=feature
                    value=v
                    max_entropy=entropy
        x=np.concatenate((x,nans_x), axis=0)#Возвращение пропусков в выборки
        y=np.concatenate((y,nans_y), axis=0)
    nans_idx=np.where(pd.isnull(x[:,feat]))
    nans_x=x[nans_idx]
    nans_y=y[nans_idx]
    x=np.delete(x, nans_idx, axis=0)
    y=np.delete(y, nans_idx, axis=0)
    X_r, X_l, y_r,y_l=split_data(x,y,feat,value)
    q=len(X_l)/(len(X_r)+len(X_l))
    rans=np.array(np.random.choice([0,1], len(nans_x), replace=True, p=[q,1-q]))#Распределение пропусков по вероятностям их попадания в ту или иную ветвь
    X_l=np.concatenate((X_l,nans_x[np.where(rans==0)]),axis=0)
    y_l=np.concatenate((y_l,nans_y[np.where(rans==0)]),axis=0)
    X_r=np.concatenate((X_r,nans_x[np.where(rans==1)]), axis=0)
    y_r=np.concatenate((y_r,nans_y[np.where(rans==1)]),axis=0)
    return feat, value, (X_r, X_l, y_r,y_l)


def sum_dicts(d_1,d_2):#Суммирование словарей по ключам
    for i in d_1.keys():
        d_1[i]+=d_2[i]
    return d_1

def get_values(rules, level, key, un): #Получение сумм всех предсказаний по листам, исходящих из вершины
    d={}
    for i in un:
        d[i]=0
    for spl in rules[level+1]:
        if spl[0]==key+'0' or spl[0]==key+'1':
            if spl[1]=='leaf':
                d[spl[2]]+=spl[3]
            else:
                d=sum_dicts(d,get_values(rules, level+1,spl[0],un))
    return d

def delete_branch(rul, index, lvl):#Удаление одной из веток узла
    indexes=[index]
    r=rul.copy()
    while lvl<=len(r)-1:
        to_delete=[]
        for ind, rule in enumerate(r[lvl]):
            if rule[0] in indexes:
                indexes.extend([rule[0]+'0', rule[0]+'1'])
                to_delete.append(ind)
        for i in list(reversed(to_delete)):
            r[lvl].pop(i)
        lvl+=1
    return r

def move_values(rul, level, ind_1,ind_2):#Перемещение ветки на один уровень выше
    indexes=[ind_2+'0',ind_2+'1']
    k='0'
    to_delete=[]
    to_next=[]
    for ind, split in enumerate(rul[level+1]):
        if split[0] in indexes:
            rul[level].append([ind_1+k, split[1], split[2],split[3]])
            if split[1]=='node':
                to_next.append([ind_1+k,split[0]]) 
            k='1'
            to_delete.append(ind)
    for i in list(reversed(to_delete)):
        rul[level+1].pop(i)
    for el in to_next:
        move_values(rul, level+1, el[0], el[1])
    return rul

def change_nodes(rules, index, level, way):#Полная замена узла на ветку
    if way==0:
        ind_to_del=index+'1'
        ind_to_move=index+'0'
    else:
        ind_to_del=index+'0'
        ind_to_move=index+'1'
    rules=delete_branch(rules, ind_to_del, level+1)#Удаление другой ветки
    if way==0:
        lvl=level+1
        while lvl<=len(rules)-1:
            for spl in rules[lvl]:
                if spl[0][:level+2]==ind_to_move:
                   spl[0]=ind_to_del+spl[0][level+2:]
            lvl+=1
        ind_to_move=ind_to_del
    for num, split in enumerate(rules[level+1]):#Меняем узел на нужную ветку
        if split[0]==ind_to_move:
            for n, spl in enumerate(rules[level]):
                if spl[0]==index:
                    ix=n
            rules[level][ix]=[index, split[1],split[2],split[3]]
            rules[level+1].pop(num)
    rules=move_values(rules,level+1,index, ind_to_move)#Сдвигаем ветку на уровень вверх
    return rules

def make_leaf(r,level, ind, un):#Превращение узла в лист
    d=get_values(r, level,ind, un)
    r=delete_branch(r, ind+'0', level+1)
    r=delete_branch(r, ind+'1', level+1)
    for i, split in enumerate(r[level]):
        if split[0]==ind:
            n=i
    r[level][n]=[ind, 'leaf', (max(zip(d.values(), d.keys()))[1]),(max(zip(d.values(), d.keys()))[0])]
    return r

def find_best_node(classifier, index, level, X_test, y_test):#Поиск лучшего варианта изменения узла при стрижки дерева
    r_1=copy.deepcopy(classifier.rules)
    r_1=change_nodes(r_1,index,level,1)
    r_2=copy.deepcopy(classifier.rules)
    r_2=change_nodes(r_2,index,level,0)
    r_3=copy.deepcopy(classifier.rules)
    r_3=make_leaf(r_3, level,index, np.unique(y_test))
    dc_1=DecisionTreeClassifier()
    dc_1.rules=r_1
    dc_2=DecisionTreeClassifier()
    dc_2.rules=r_2
    dc_3=DecisionTreeClassifier()
    dc_3.rules=r_3
    predicts=[np.sum(classifier.predict(X_test)==y_test)/len(y_test),np.sum(dc_1.predict(X_test)==y_test)/len(y_test),
              np.sum(dc_2.predict(X_test)==y_test)/len(y_test),np.sum(dc_3.predict(X_test)==y_test)/len(y_test)]
    if predicts[1]==max(predicts):
        return r_1
    elif predicts[2]==max(predicts):
        return r_2
    elif predicts[3]==max(predicts):
        return r_3
    else:
        return classifier.rules

def prune(tree, X_val, y_val):#Стрижка дерева
    level=0
    index='0'
    d={'0':[X_val,y_val]}
    while level<=len(tree.rules)-1:
        c=0
        while c<=len(tree.rules[level])-1:
            if tree.rules[level][c][1]=='node':
                if len(d.get(tree.rules[level][c][0])[0])==0:
                    tree.rules=make_leaf(tree.rules,level,tree.rules[level][c][0], tree.uniques)
                else:
                    tree.rules=find_best_node(tree,tree.rules[level][c][0],level,X_val,y_val)
                    if tree.rules[level][c][1]=='node':
                        X_l,X_r,y_l,y_r=split_data(d.get(tree.rules[level][c][0])[0],d.get(tree.rules[level][c][0])[1],tree.rules[level][c][2],tree.rules[level][c][3])
                        d[tree.rules[level][c][0]+'0']=[X_l,y_l]
                        d[tree.rules[level][c][0]+'1']=[X_r,y_r]
            c+=1
        level+=1
    to_del=[]
    for i, level in enumerate(tree.rules):
        if len(level)==0:
            to_del.append(i)
    for i in list(reversed(to_del)):
        tree.rules.pop(i)
    return tree.rules


class DecisionTreeClassifier():#Дерево классификации
    rules=[]#Массив из правил
    splits=[]#Массив из разбиений
    uniques=[]#Массив из уникальных значений
    def fit(self, x,y, crit='e'):
        self.uniques=np.unique(y)
        splits=[[['0',x,y]]]
        rules=[]
        k=1
        while k==len(splits):#Пока дерево растёт в глубину, продолжаем
            p=[]
            r=[]
            for split in splits[k-1]:#Рассматриваем каждый уровень дерева
                u=np.unique(split[2])
                if len(u)==1:#Если всего один уникальный класс, то делаем лист
                    r.append([split[0], 'leaf', u[0], len(split[2])])
                else:#Иначе - ищем оптимальное разбиение, делаем узел и 2 разбиения на следующем уровне
                    feature, value, [X_r, X_l, y_r,y_l]= get_optimal_split(split[1],split[2], crit)
                    p.append([split[0]+'0',X_l,y_l])
                    p.append([split[0]+'1',X_r,y_r])
                    r.append([split[0], 'node', feature, value])
            rules.append(r)
            k+=1
            if len(p)>0:
                splits.append(p)
        self.splits=splits
        self.rules=rules
    def get_all_values(self, level, key):#Получение суммарного количества объектов каждого класса в листах уровнями ниже
        d={}
        for i in self.uniques:
            d[i]=0
        for spl in self.rules[level+1]:
            if spl[0]==key+'0' or spl[0]==key+'1':
                if spl[1]=='leaf':
                    d[spl[2]]+=spl[3]
                else:
                    d=sum_dicts(d,self.get_all_values(level+1,spl[0]))
        return d

        
    def predict(self,x):
        predict=[]
        for i in range(len(x)):
            xt=x[i,:]
            k_start='0'
            level=0
            flg=True
            while flg==True:
                for m in self.rules[level]:
                    if m[0]==k_start:
                        if m[1]=='leaf':#Если попали в лист, то выдаём класс, в который попали
                            predict.append(m[2])
                            flg=False
                        elif pd.isnull(xt[m[2]])==True:#Если пустое значение, то выдаём наиболее вероятное значение
                            d=self.get_all_values(level, k_start)
                            predict.append(max(zip(d.values(), d.keys()))[1])
                            flg=False
                        else:#Если попали в узел, то по предикату идём на следующий уровень
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