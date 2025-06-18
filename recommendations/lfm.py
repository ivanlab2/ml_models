import numpy as np

class LFM:
    def __init__(self, n_users, n_movies, n_factors):
        self.n_users = n_users#Число пользователей
        self.n_movies = n_movies#Число фильмов 
        self.n_factors = n_factors#Число факторов
        #Начальная инициализация матриц пользователей и фильмов, а также bias-ов
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_movies, n_factors))
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_movies, dtype=np.float32)
    def fit(self, R,  n_iters=100, alpha=0.01, lam=0.1, eps=0.001):
        losses=[]
        for _ in range(n_iters):#Стохастический градиентный спуск
            np.random.shuffle(R)
            mse=0
            avg_loss=99999
            for d in R:
                u,i,r=d.astype(int)
                pred = self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])#Предсказание
                eui = r - pred
                #Обновление весов
                self.b_u[u] += alpha * (eui - lam * self.b_u[u])
                self.b_i[i] += alpha * (eui - lam * self.b_i[i])
                self.P[u] += alpha * (eui * self.Q[i] - lam * self.P[u])
                self.Q[i] += alpha * (eui * self.P[u] - lam * self.Q[i])
                mse+=eui**2
            avg_loss_new=mse/R.shape[0]
            #Остановка при сходимости
            if abs(avg_loss_new-avg_loss)<eps:
                break
            else:
                avg_loss=avg_loss_new
                losses.append(avg_loss) 
        return losses
    def predict(self, R):
        y_pred=np.array([])
        for d in R:
            u,i,_=d.astype(int)
            y_pred=np.append(y_pred,self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i]))
        return y_pred
    


