import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import serial
import time
# khoi tạo ket noi arduino
DataSerial = serial.Serial('com6', 115200)
time.sleep(5)
# dong mo servo
def mo():
    DataSerial.write('mo\r'.encode())

def dong():
    DataSerial.write('dong\r'.encode())

# Đường dẫn đến file mô hình đã được huấn luyện
model_path = "cnn_model"

# Kích thước ảnh đầu vào
image_width, image_height = 250, 250

# Load mô hình đã được huấn luyện
model = tf.keras.models.load_model(model_path)

# Chúng ta cần khai báo lại class_labels để tương ứng với số lớp
class_labels = {0: "co_tau", 1: "khong_co_tau"}

# Khởi tạo camera
camera = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ camera
    ret, frame = camera.read()

    if not ret:
        break

    # Chuyển đổi khung hình thành ảnh RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Thay đổi kích thước ảnh
    image = cv2.resize(image, (image_width, image_height))

    # Chuẩn hóa ảnh
    image = image / 255.0

    # Mở rộng kích thước ảnh thành batch size = 1
    image = np.expand_dims(image, axis=0)

    # Dự đoán lớp của ảnh
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]

    # Hiển thị kết quả dự đoán lên khung hình
    cv2.putText(frame, predicted_class, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Camera", frame)

    print(predicted_class)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if predicted_class == 'co_tau':
        dong()
    elif predicted_class == 'khong_co_tau':
        mo()

# Giải phóng camera và đóng cửa sổ hiển thị
camera.release()
cv2.destroyAllWindows()