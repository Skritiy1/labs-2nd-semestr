import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
import random


random_numbers = random.sample(range(1, 12), 3)
print("Выбранные номера методов:", random_numbers)



def generate_data_1(n_samples=300):
    """Кластеры с разной плотностью"""
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=[0.4, 0.5, 0.2])
    return X


def generate_data_2(n_samples=300):
    """Два полумесяца"""
    X, _ = make_moons(n_samples=n_samples, noise=0.05)
    return X


def generate_data_3(n_samples=300):
    """Концентрические окружности"""
    X, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
    return X


def generate_data_4(n_samples=300):
    """Анизотропно распределенные данные"""
    X, _ = make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
    return X


def generate_data_5(n_samples=300):
    """Кластеры с переменной дисперсией"""
    X, _ = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])
    return X


def generate_data_6(n_samples=300):
    """Случайные данные без четкой структуры"""
    X, _ = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=4)
    return X


def cluster_dbscan(X):
    """Кластеризация с помощью DBSCAN"""
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    return db.labels_


def cluster_meanshift(X):
    """Кластеризация с помощью MeanShift"""
    X = StandardScaler().fit_transform(X)
    ms = MeanShift(bandwidth=0.5).fit(X)
    return ms.labels_


def cluster_agglomerative(X):
    """Кластеризация с помощью AgglomerativeClustering"""
    X = StandardScaler().fit_transform(X)
    ac = AgglomerativeClustering(n_clusters=3).fit(X)
    return ac.labels_


data_generators = [
    generate_data_1,
    generate_data_2,
    generate_data_3,
    generate_data_4,
    generate_data_5,
    generate_data_6
]

cluster_methods = [
    ("DBSCAN", cluster_dbscan),
    ("MeanShift", cluster_meanshift),
    ("Agglomerative", cluster_agglomerative)
]


plt.figure(figsize=(15, 20))
plot_num = 1

for i, generate_data in enumerate(data_generators):
    X = generate_data()
    X = StandardScaler().fit_transform(X)

    for j, (name, cluster_func) in enumerate(cluster_methods):
        plt.subplot(len(data_generators), len(cluster_methods), plot_num)


        labels = cluster_func(X)

        # Визуализация
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
        plt.title(f"Data {i + 1} - {name}")
        plt.xticks([])
        plt.yticks([])

        plot_num += 1

plt.tight_layout()
plt.show()

# Анализ
print("""
Анализ результатов:
1. DBSCAN:
   - Хорошо работает с кластерами произвольной формы (полумесяцы, окружности)
   - Плохо работает с анизотропными данными и кластерами разной плотности
   - Автоматически определяет количество кластеров

2. MeanShift:
   - Хорошо работает с кластерами одинаковой плотности
   - Плохо работает с кластерами разной плотности и сложной формы
   - Автоматически определяет количество кластеров

3. AgglomerativeClustering:
   - Хорошо работает с компактными кластерами
   - Плохо работает с кластерами сложной формы (полумесяцы, окружности)
   - Требует указания количества кластеров
""")