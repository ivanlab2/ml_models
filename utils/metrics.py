from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import r2_score


def mse(y_pred, y):#Функция потрь - MSE
  err = np.mean(np.square(y_pred - y))
  return err

def accuracy(y_pred,y_test):#Точность
    return np.sum(y_pred==y_test)/len(y_test)

def create_model(model_class, **init_kwargs):
    def builder():
        return model_class(**init_kwargs)
    return builder


def cross_validation(model_builder, X, y, n_splits=5, fit_params=None, fit_predict=False, regression=False, mse_metric=False):
    """
    model_class: класс модели или функция, возвращающая объект модели
    fit_params: словарь с параметрами для метода fit
    """
    if fit_params is None:
        fit_params = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kf.split(X):
        model = model_builder()  # создаём новую модель на каждую итерацию
        if fit_predict==True:
            y_pred=model.fit_predict(X[train_idx], y[train_idx],X[test_idx],**fit_params)
        else:
            model.fit(X[train_idx], y[train_idx], **fit_params)
            y_pred = model.predict(X[test_idx])
        if regression==True:
            if mse_metric==False:
                accs.append(r2_score(y[test_idx],y_pred))
            else:
               accs.append(mse(y_pred, y[test_idx])) 
        else:
            accs.append(accuracy(y_pred, y[test_idx]))
    return np.mean(accs)


def get_topics_words(phi, feature_names, top_n=10):#Получение топ-слов по каждой теме
    return [
        [feature_names[i] for i in topic.argsort()[-top_n:][::-1]]
        for topic in phi
    ]

def get_coherence_score(vectorizer, phi, texts, metric='u_mass'):#Расчёт когерентности тем
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    feature_names = vectorizer.get_feature_names_out()
    topics = get_topics_words(phi, feature_names, top_n=10)#Получение топ-слов
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in texts]#Получение токенов
    dictionary = Dictionary(tokens)#Получение словаря

    
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokens,
        dictionary=dictionary,
        coherence=metric
    )
    coherence_score = coherence_model.get_coherence()#Расчёт когерентности
    print(f"Когерентность тем ({metric}): {coherence_score:.4f}")

def euclide(x,y):#Расчёт евклидового расстояния между двумя точками
    return np.sqrt(np.sum(np.power(x-y,2)))

def max_n_em(A,data, i_1,i_2):#Метод дальнего соседа для расчёта среднего межкластерного расстояния
    m=-1
    for i in A[i_1][0]:
        for j in A[i_2][0]:
            t=euclide(data[i],data[j])
            if m<t:
                m=t
    return m

def cross_validation_with_pruning(model_builder, X, y, X_val,y_val, n_splits=5, fit_params=None):
    """
    model_class: класс модели или функция, возвращающая объект модели
    fit_params: словарь с параметрами для метода fit
    """
    from trees.ClassifierTree import prune
    from trees.RegressionTree import prune_r
    if fit_params is None:
        fit_params = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=204)
    accs = []

    for train_idx, test_idx in kf.split(X):
        model = model_builder()  # создаём новую модель на каждую итерацию
        model.fit(X[train_idx], y[train_idx], **fit_params)
        model.rules=prune(model, X_val,y_val)
        y_pred = model.predict(X[test_idx])
        accs.append(accuracy(y_pred, y[test_idx]))
    return np.mean(accs)




