from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def data_scientist_part():
    # Загрузка датасета digits (предполагаем, что это датасет 6/10)
    data = load_digits()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['target'])

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Вывод информации о данных
    print("Информация о датасете:")
    print(f"Общее количество образцов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}\n")

    # Вывод примеров данных
    print("Обучающая выборка (первые 3 элемента):")
    print(X_train.head(3))
    print("\nТестовая выборка (первые 3 элемента):")
    print(X_test.head(3))

    return X_train, X_test, y_train, y_test


# Выполнение части специалиста по данным
X_train, X_test, y_train, y_test = data_scientist_part()

# =============================================
# Часть 2: Разработчик (Developer)
# =============================================
print("\n" + "=" * 50)
print("Часть 2: Выполняет разработчик")
print("=" * 50 + "\n")


def developer_part(X_train, X_test, y_train, y_test):
    # Преобразуем y в 1D array
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_knn)
    precision = precision_score(y_test, y_pred_knn, average='weighted')
    recall = recall_score(y_test, y_pred_knn, average='weighted')

    print(f"Accuracy KNN: {accuracy:.4f}")
    print(f"Precision KNN: {precision:.4f}")
    print(f"Recall KNN: {recall:.4f}")
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\nRandom Forest Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted'):.4f}")
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    return y_pred_knn, y_pred_rf


# =============================================
# Часть 3: Дизайнер/Визуализатор (Designer)
# =============================================
print("\n" + "=" * 50)
print("Часть 3: Выполняет дизайнер")
print("=" * 50 + "\n")


def show_examples(model_name, X_test, y_test, y_pred, n_samples=5):
    # Преобразуем в numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Получаем индексы правильных и неправильных классификаций
    correct = np.where(y_test == y_pred)[0]
    incorrect = np.where(y_test != y_pred)[0]

    # Выбираем случайные примеры
    np.random.shuffle(correct)
    np.random.shuffle(incorrect)

    # Выбираем первые n_samples примеров
    correct = correct[:n_samples]
    incorrect = incorrect[:min(n_samples, len(incorrect))]

    # Создаем фигуру для отображения примеров
    plt.figure(figsize=(15, 3 * (n_samples + 1)))
    plt.suptitle(f'Примеры классификации ({model_name})', fontsize=16)

    # Отображаем правильные классификации
    plt.subplot(2, 1, 1)
    plt.title('Верные классификации')
    for i, idx in enumerate(correct):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}')
        plt.axis('off')

    # Отображаем неправильные классификации
    plt.subplot(2, 1, 2)
    plt.title('Ошибочные классификации')
    for i, idx in enumerate(incorrect):
        plt.subplot(2, n_samples, n_samples + i + 1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


y_pred_knn, y_pred_rf = developer_part(X_train, X_test, y_train, y_test)

# Визуализация результатов
show_examples("KNN", X_test, y_test, y_pred_knn)
show_examples("Random Forest", X_test, y_test, y_pred_rf)