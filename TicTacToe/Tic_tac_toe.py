import csv
import random
from src.naiveBayesClassifier import NaiveBayesClassifier, cross_validation_accuracy

train_file_path = 'Tic_tac_toe.txt'

with open(train_file_path, mode='r') as train:

    ttt_data = csv.reader(train)

    ttt_data = [row for row in ttt_data]

    split_ratio = 0.1

    x = []
    y = []
    for i in range(85):
        acc = cross_validation_accuracy(NaiveBayesClassifier(9), ttt_data, split_ratio, 100)
        print(acc)
        x.append(split_ratio)
        y.append(acc)
        split_ratio += 0.01

    import matplotlib.pyplot as plt

    # Построение графика
    plt.plot(x, y, label='Линия 1')

    # Добавление подписей осей и заголовка
    plt.xlabel('Доля данных для обучения')
    plt.ylabel('Точность модели')
    plt.title('Зависимость точности модели от доли данных для обучения')

    # Добавление легенды
    plt.legend()

    # Отображение графика
    plt.show()