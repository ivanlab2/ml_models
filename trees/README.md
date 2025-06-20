# Decision trees: Classification & Regression
Данный модуль содержит реализации деревьев решений методом ID3:

- **Дерево классификации**
- **Дерево регрессии**

## Описание моделей

### Дерево классификации

- Реализован классический алгоритм классификации
- Реализован алгоритм **прунинга**
- Дерево обрабатывает **категориальные** и **регрессионные** признаки
- Обработка **пропусков**
- Использоваие критериев **Донского** и **энтропии** разбиения

### Дерево регрессии

- Реализован классический алгоритм регрессионного дерева
- Дерево обрабатывает категориальные и регрессионные признаки
- Обработка пропусков
- Использование MSE-критерия разбиения

## Данные

Для тестирования дерева классификации использовался датасет о предсказании погоды с Kaggle:

🔗 [Weather Classification](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification)

Для тестирования дерева регрессии использовался датасет о предсказании стоимости страховки с Kaggle:

🔗 [Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Важно:** после скачивания положите файлы в папку `data`.

## Метрики качества


| Модель               | Accuracy | Время обучения |
|----------------------|-----------------------------|---------------------|
| Собственная реализация (критерий - энтропия)        | 0.8862                     | 44918.757 ms             |
| Собственная реализация (критерий - Донского)        | 0.8873                     | 45183.79 ms             |
| Собственная реализация (критерий - энтропия, pruning)       |  0.905                     | 29.51 ms             |
| Эталонная реализация  (критерий - энтропия)    | 0.8821                     | -            |


| Модель               | R^2 | Время обучения |
|----------------------|-----------------------------|---------------------|
| Собственная реализация       | 0.8178                    | 4149.323 ms           |
| Эталонная реализация     | 0.8052                     | 1.998 ms             |




## Примечание

- Эталонная реализация взята из `scikit-learn` и использована для сравнения производительности и точности.
- В `scikit-learn` не был реализован критерий Донского, а также не предоставлена возможность обрабатывать категориальный признаки без предобработки. Соответственно, сравниваемые модели не являются идентичными.
- Более подробные эксперименты и визуализации представлены в ноутбуке: [`notebooks/trees_test.ipynb`](../notebooks/trees_test.ipynb).
