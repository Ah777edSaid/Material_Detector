import cv2
import tensorflow as tf
import numpy as np
import requests

def download_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:  # Check if the request was successful
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully and saved to {save_path}")
    else:
        print("Failed to retrieve the image")

# download_image
image_url = "https://th.bing.com/th/id/OIP.VCZorWbps0gxQVPO9yOJawHaFj?rs=1&pid=ImgDetMain"  # Replace with your image URL
save_path = "preprocess_image.jpg"  # Path to save the image
download_image(image_url, save_path)


# تحميل نموذج MobileNet المدرب مسبقًا
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# دالة لمعالجة الصورة وتغيير حجمها لتناسب المدخلات
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # تحويل BGR إلى RGB
    image = cv2.resize(image, (224, 224))  # تغيير الحجم ليكون 224x224 لتوافق MobileNet
    image = np.expand_dims(image, axis=0)  # إضافة بعد إضافي ليصبح الشكل (1, 224, 224, 3)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # تجهيز البيانات
    return image

# دالة للتنبؤ بالنوع باستخدام النموذج
def predict_material(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    
    print(f"Predicted material: {decoded_predictions[1]} with confidence {decoded_predictions[2]:.2f}")
    return decoded_predictions

# تحديد نوع المادة
image_path = '/content/preprocess_image.jpg'  # أدخل مسار الصورة هنا
predict_material(image_path)
