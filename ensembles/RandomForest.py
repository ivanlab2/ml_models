import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils.metrics import accuracy

def find_most_frequent(preds):#Определение предсказания методом голосования
    for i in range(len(preds)):
        a,b=np.unique(preds,return_counts=True)
        return a[np.argmax(b)]
    
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