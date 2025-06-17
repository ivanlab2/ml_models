
import numpy as np
import matplotlib.pyplot as plt

class RidgeRegressor:
    def __init__(self, fit_intercept):
        self.intercept=fit_intercept#Наличие свободного члена
    def fit(self,X,y, t=0):#Получение SVD-матриц, высчитывание вектора весов
        if self.intercept==True:
            X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)#Если есть свободный член, то добавляем слева столбец с единицами
        self.t=t#Параметр регуляризации
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)  #SVD-разложение X
        S_inv_reg = self.S / (self.S ** 2 + self.t)  #Расчёт регуляризованной диагональная матрица
        self.W = self.V.T @ np.diag(S_inv_reg) @ self.U.T @ y  #Расчёт весов
    def fit_just_weights(self,y, t): #Перерасчитывание весов без изменения SVD-матрицы 
        self.t = t
        S_inv_reg = self.S / (self.S ** 2 + self.t)
        self.W = self.V.T @ np.diag(S_inv_reg) @ self.U.T @ y
    def count_Q(self, X_val, y_train, y_val, t): #Высчитывание функционала потерь на контрольной выборке
        if self.intercept:
            X_val = np.concatenate((np.ones((X_val.shape[0], 1)), X_val), axis=1)
        S_inv_reg = self.S / (self.S ** 2 + t)
        W_tmp = self.V.T @ np.diag(S_inv_reg) @ self.U.T @ y_train
        y_pred = X_val @ W_tmp
        return np.linalg.norm(y_pred - y_val)
    def choose_t(self, X_val,y_train,y_val,t_start,t_end,step):#Подбор оптимального параметра регуляризации
        Q_s=[]
        for i in range(t_start,t_end,step):
            Q_s.append(self.count_Q(X_val, y_train, y_val, i)) #Для каждого значения параметра считается функционал потерь
        plt.plot(range(t_start,t_end,step),Q_s)
        plt.title('Зависимость функционала ошибки от параметра регуляризации')
        plt.xlabel('Значение параметра регуляризации')
        plt.ylabel('Значение функции потерь')
        plt.grid()
        plt.show()
    def cond_number(self):#Расчёт числа обусловленности матрицы
        return np.max(self.S)/np.min(self.S)
    def predict(self, X):#Предсказание (умножение входящей матрицы на рассчитанные весы)
        if self.intercept==True:
            return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)@self.W
        else:
            return X@self.W