import numpy as np
from sklearn.model_selection import train_test_split

def min_max_scale(X):
    mins_x=np.min(X,axis=0)
    maxs_x=np.max(X,axis=0)
    X_st=(X-mins_x)/(maxs_x-mins_x)
    return X_st

def standartize_data(data):#Стандартизация данных
    d=(data-data.mean())/data.std()
    return d.to_numpy()

def standartize_data_to_matrix(data): #Стандартизация данных
    d=(data-data.mean())/data.std()
    return np.asmatrix(d)

def prepare_recommendation_data(df):
    #Получение индексов уникальных пользователей и фильмов
    unique_users = sorted(df['userId'].unique())
    unique_items = sorted(df['movieId'].unique())
    #Упорядочивание пользователей и фильмов от 0 до n
    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
    df['user_idx'] = df['userId'].map(user_id_map)
    df['item_idx'] = df['movieId'].map(item_id_map)
    R= df[['user_idx', 'item_idx', 'rating']].to_numpy()#Возвращение обновленной таблицы
    R_train, R_test = train_test_split(R, test_size=0.2, random_state=42)#Разделение на тренировочную и тестовую выборки
    return R_train, R_test, len(user_id_map), len(item_id_map)

def split_data(x,y,crit,value):#Функция разделения данных для деревьев решений
    try: 
        x[:,crit].astype('float64')
        X_r=x[np.where(x[:,crit]>=value)]
        X_l=x[np.where(x[:,crit]<value)]
        y_r=y[np.where(x[:,crit]>=value)]
        y_l=y[np.where(x[:,crit]<value)]
    except ValueError:
        X_r=x[np.where(x[:,crit]==value)]
        X_l=x[np.where(x[:,crit]!=value)]
        y_r=y[np.where(x[:,crit]==value)]
        y_l=y[np.where(x[:,crit]!=value)]
    return X_r, X_l, y_r,y_l