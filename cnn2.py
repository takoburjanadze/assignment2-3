pip install Pillow
pip install requests
from PIL import Image
import requests
from io import BytesIO

\def resize_image(image, size=(32, 32)):
    return image.resize(size)

image_url = "https://images.pexels.com/photos/45851/bird-blue-cristata-cyanocitta-45851.jpeg?auto=compress&cs=tinysrgb&w=600"

response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

resized_image = resize_image(image)

resized_image.save("bird.png")
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

new_image_path = 'bird.png'
new_image = tf.keras.preprocessing.image.load_img(new_image_path, target_size=(32, 32))
new_image_array = tf.keras.preprocessing.image.img_to_array(new_image)
new_image_array = tf.expand_dims(new_image_array, 0)  # Add batch dimension
new_image_array = new_image_array / 255.0  # Normalize pixel values

predictions = model.predict(new_image_array)
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

print("Predicted class:", predicted_class)
