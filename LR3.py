import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from random import uniform


# Выбор исходной функции (линейная регрессия)
def linear_func(x, k, b):
    return k * x + b


# Генерация исходных данных
x_min = -10
x_max = 10
points = 50
x = np.linspace(x_min, x_max, points)

# Истинные параметры
true_k = 2.5
true_b = -1.0

# Генерация y с шумом
y = linear_func(x, true_k, true_b) + np.array([uniform(-3, 3) for _ in range(points)])


# Функции для вычисления частных производных
def get_dk(x, y, k, b):
    n = len(x)
    return (2 / n) * sum(x * (k * x + b - y))


def get_db(x, y, k, b):
    n = len(x)
    return (2 / n) * sum(k * x + b - y)


# Инициализация параметров для градиентного спуска
speed = 0.001  # скорость обучения
epochs = 1000  # количество итераций
k0 = 0.0  # начальное значение k
b0 = 0.0  # начальное значение b



def fit(x, y, speed, epochs, k0, b0):
    k = k0
    b = b0
    k_history = [k]
    b_history = [b]

    for _ in range(epochs):
        dk = get_dk(x, y, k, b)
        db = get_db(x, y, k, b)

        k = k - speed * dk
        b = b - speed * db

        k_history.append(k)
        b_history.append(b)

    return k, b, k_history, b_history


# Обучение модели
final_k, final_b, k_history, b_history = fit(x, y, speed, epochs, k0, b0)


fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

# Исходные точки
scatter = ax.scatter(x, y, color='blue', label='Исходные данные')


line, = ax.plot(x, linear_func(x, k_history[0], b_history[0]), 'r-', label='Регрессия')


ax.plot(x, linear_func(x, true_k, true_b), 'g--', label='Истинная функция')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Линейная регрессия с градиентным спуском')
ax.legend()
ax.grid(True)


ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Эпоха', 0, epochs, valinit=0, valstep=1)



def update(val):
    epoch = int(slider.val)
    current_k = k_history[epoch]
    current_b = b_history[epoch]
    line.set_ydata(linear_func(x, current_k, current_b))
    fig.canvas.draw_idle()


slider.on_changed(update)


text_box = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top')


def update_text(epoch):
    current_k = k_history[epoch]
    current_b = b_history[epoch]
    text_box.set_text(f'Эпоха: {epoch}/{epochs}\nk = {current_k:.4f}\nb = {current_b:.4f}')


update_text(0)
slider.on_changed(lambda val: update_text(int(val)))

plt.show()


print(f"Истинные параметры: k = {true_k}, b = {true_b}")
print(f"Найденные параметры: k = {final_k:.4f}, b = {final_b:.4f}")


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


y_pred = linear_func(x, final_k, final_b)
print(f"MSE: {mse(y, y_pred):.4f}")