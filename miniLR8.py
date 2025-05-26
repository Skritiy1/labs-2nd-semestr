import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import random


# определяем исходную функцию
def f(x):
    return np.sin(x) + 0.5 * x ** 2 - 3 * np.log(x + 1)


# 2. Генерируем данные
x_min, x_max = 1, 10
x = np.linspace(x_min, x_max, 100).reshape(-1, 1)
e = np.array([random.uniform(-3, 3) for _ in range(100)])
y = f(x).flatten() + e

# создаем и обучаем модели регрессии
models = {
    "Линейная регрессия": LinearRegression(),
    "SVR (ядро RBF)": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    "Случайный лес": RandomForestRegressor(n_estimators=100, random_state=42)
}

# словарь для хранения MSE
results = {}

# обучаем модели, делаем предсказания и строим графики
plt.figure(figsize=(18, 12))
for i, (name, model) in enumerate(models.items(), 1):
    # Обучение и предсказание
    model.fit(x, y)
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    results[name] = mse

    # Построение графика
    plt.subplot(2, 2, i)
    plt.scatter(x, y, color='blue', label='Исходные точки', alpha=0.5)
    plt.plot(x, f(x), color='green', label='Исходная функция', linewidth=2)
    plt.plot(x, y_pred, color='red', label=f'Предсказание ({name})', linewidth=2)
    plt.title(f'{name}\nMSE: {mse:.2f}')
    plt.legend()
    plt.grid(True)

# общий график для сравнения
plt.subplot(2, 2, 4)
plt.scatter(x, y, color='blue', label='Исходные точки', alpha=0.5)
plt.plot(x, f(x), color='green', label='Исходная функция', linewidth=2)
for name, model in models.items():
    plt.plot(x, model.predict(x), '--', label=f'Предсказание ({name})', linewidth=1.5)
plt.title('Сравнение всех методов')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# вывод результатов в терминал
print("\nРезультаты сравнения методов регрессии:")
print("-------------------------------------")
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.2f}")

# анализ результатов
best_model = min(results, key=results.get)
worst_model = max(results, key=results.get)

print("\nВыводы:")
print("1. Наилучшие результаты показывает", best_model,
      f"с MSE = {results[best_model]:.2f}. Этот метод лучше всего аппроксимирует",
      "нелинейную природу данных.")

print("2. Наихудшие результаты у", worst_model,
      f"с MSE = {results[worst_model]:.2f}.", end=' ')
if worst_model == "Линейная регрессия":
    print("Это ожидаемо, так как линейная модель не может адекватно описать",
          "трансцендентно-алгебраическую функцию.")
else:
    print("Возможно, требуется настройка гиперпараметров для улучшения результатов.")

print("3. SVR показывает хорошие результаты благодаря использованию RBF ядра,",
      "которое хорошо подходит для нелинейных зависимостей.")

print("4. Случайный лес демонстрирует хорошую точность, но может быть",
      "чувствителен к переобучению при увеличении глубины деревьев.")