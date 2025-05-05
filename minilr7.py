import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Генерация данных
def generate_datasets(n_samples=500, seed=30):

    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)


    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)


    cluster_std = [1.0, 0.5]
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=cluster_std, random_state=seed, centers=2)


    random_state = 170
    x, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, centers=2)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    x_aniso = np.dot(x, transformation)
    aniso = (x_aniso, y)


    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, centers=2)

    return [noisy_circles, noisy_moons, varied, aniso, blobs]


# cоздание моделей
def create_models():

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')


    tree_model = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)


    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                             solver='adam', max_iter=1000, random_state=42)

    return svm_model, tree_model, nn_model


# Визуализация результатов
def plot_results(models, datasets, model_names):
    plt.figure(figsize=(18, 25))

    for i, (X, y) in enumerate(datasets):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        for j, (name, model) in enumerate(zip(model_names, models)):
            plt.subplot(len(datasets), len(models), i * len(models) + j + 1)


            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)


            if hasattr(model, "predict_proba"):
                Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            else:
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)


            accuracy = accuracy_score(y_test, y_pred)


            plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')


            for k in range(len(X_train)):
                plt.scatter(X_train[k, 0], X_train[k, 1],
                            marker='x' if y_train[k] == 0 else 'o',
                            c='blue', alpha=0.5)


            for k in range(len(X_test)):
                color = 'green' if y_test[k] == y_pred[k] else 'red'
                plt.scatter(X_test[k, 0], X_test[k, 1],
                            marker='x' if y_test[k] == 0 else 'o',
                            c=color, edgecolors='black')

            plt.title(f'{name}\nDataset {i + 1}, Accuracy: {accuracy:.2f}')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    datasets = generate_datasets()


    svm_model, tree_model, nn_model = create_models()
    models = [svm_model, tree_model, nn_model]
    model_names = ['SVM', 'Decision Tree', 'Neural Network (MLP)']


    plot_results(models, datasets, model_names)