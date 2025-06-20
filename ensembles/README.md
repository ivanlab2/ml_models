# Ensemble Models: Random Forest and Gradient Boosting

Данный модуль содержит две реализованные ансамблевые модели машинного обучения:

- **Случайный лес** (бинарный классификатор)
- **Градиентный бустинг** (бинарный классификатор)

## Описание моделей

### Случайный лес

- Бинарный классификатор на основе ансамбля деревьев.
- В реализацию добавлена фильтрация: в ансамбль включаются только те деревья, которые достигают заданной точности (accuracy) на **тренировочной** и **отложенной** выборках.

### Градиентный бустинг

- Классификатор с использованием **экспоненциальной функции потерь**.
- Реализована возможность построения графика изменения точности модели по мере увеличения числа деревьев в ансамбле.

## Данные

Для тестирования использовался датасет о раке груди с Kaggle:

🔗 [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

**Важно:** после скачивания положите файл в папку `data`.

## 📊 Метрики качества

| Модель               | Accuracy (своя реализация) | Accuracy (sklearn) | Время (своя реализация) | Время (sklearn) |
|----------------------|----------------------------|---------------------|--------------------------|------------------|
| Градиентный бустинг  | 0.9437                     | 0.9437              | 289.130 ms               | 229.047 ms       |
| Случайный лес        | 0.9367                     | 0.9367              | 114.188 ms               | 127.318 ms       |

## Примечание

- Эталонные реализации взяты из `scikit-learn` и использованы для сравнения производительности и точности.
- Более подробные эксперименты и визуализации представлены в ноутбуке: [`notebooks/ensembles_test.ipynb`](../notebooks/ensembles_test.ipynb).
