import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

# 1. Загружаем и подготавливаем данные (картинки 28x28 пикселей)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Делаем цвета от 0 до 1 (делим на 255) и добавляем размерность канала
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Создаем простую нейросеть
model = Sequential([
    # Поиск узоров и линий на картинке
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),  # Уменьшаем картинку в 2 раза

    # Превращаем в плоский список и делаем вывод
    Flatten(),
    Dense(64, activation='relu'),  # Скрытый слой
    Dense(10, activation='softmax')  # 10 вариантов ответа (цифры от 0 до 9)
])

# 3. Настраиваем обучение
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Обучаем сеть (всего 3 эпохи для скорости)
print("Начало обучения...")
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=3, batch_size=128)

# 5. Проверяем, как она работает на тестах
print("\nПроверка работы сети:")
loss, accuracy = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
print(f"Точность сети: {accuracy * 100:.2f} %")