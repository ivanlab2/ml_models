import numpy as np
from sklearn.model_selection import train_test_split

def min_max_scale(X):
    mins_x=np.min(X,axis=0)
    maxs_x=np.max(X,axis=0)
    X_st=(X-mins_x)/(maxs_x-mins_x)
    return X_st

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