import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Đường dẫn tới tệp hình ảnh cần dự đoán
image_path = "D:\CNN\pyimagesearch/anhtest/tau.jfif"

# Đường dẫn đến file mô hình đã được huấn luyện
model_path = "cnn_model"

# Kích thước ảnh đầu vào
image_width, image_height = 250, 250

# Load mô hình đã được huấn luyện
model = tf.keras.models.load_model(model_path)

# Load và chuẩn hóa hình ảnh
image = load_img(image_path, target_size=(image_width, image_height))
image = img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Dự đoán lớp của hình ảnh
predictions = model.predict(image)
predicted_class_index = np.argmax(predictions[0])

# Chúng ta cần khai báo class_labels để tương ứng với số lớp
class_labels = {0: "co_sung", 1: "co_tau"}

# In kết quả
predicted_class = class_labels[predicted_class_index]
print("Lớp dự đoán cho hình ảnh:", predicted_class)