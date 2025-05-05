import numpy as np
import matplotlib.pyplot as plt


def gradientDescend(func=lambda x: x ** 2, diffFunc=lambda x: 2 * x,
                    x0=3, speed=0.01, epochs=100):
    """
    Реализация метода градиентного спуска
    """
    xList = []
    yList = []
    x = x0

    for _ in range(epochs):
        xList.append(x)
        yList.append(func(x))
        x = x - speed * diffFunc(x)

    return xList, yList


# Наша функция и её производная
def my_func(x):
    return x ** 2 + 3 * np.sin(x)


def my_func_derivative(x):
    return 2 * x + 3 * np.cos(x)



x_vals, y_vals = gradientDescend(func=my_func, diffFunc=my_func_derivative,
                                 x0=1, speed=0.1, epochs=50)

# Построение графика
x_plot = np.linspace(-3, 3, 400)
y_plot = my_func(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = x² + 3sin(x)')
plt.scatter(x_vals, y_vals, color='red', label='Точки градиентного спуска')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Градиентный спуск для функции f(x) = x² + 3sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Проверка сходимости
final_x = x_vals[-1]
final_y = y_vals[-1]
print(f"Финальная точка: x = {final_x:.4f}, y = {final_y:.4f}")

approx_min_x = -0.6
if abs(final_x - approx_min_x) < 0.1:
    print("Результат сходится к искомому минимуму!")
else:
    print("Результат не сходится к искомому минимуму.")



def find_critical_speed(func, diffFunc, x0=1, epochs=50, tol=0.1):
    low = 0.01
    high = 1.0
    approx_min_x = -0.6

    for _ in range(20):
        mid = (low + high) / 2
        x_vals, _ = gradientDescend(func=func, diffFunc=diffFunc,
                                    x0=x0, speed=mid, epochs=epochs)
        final_x = x_vals[-1]

        if abs(final_x - approx_min_x) < tol:
            low = mid
        else:
            high = mid

    return (low + high) / 2


critical_speed = find_critical_speed(my_func, my_func_derivative)
print(f"Граничное значение speed: {critical_speed:.4f}")