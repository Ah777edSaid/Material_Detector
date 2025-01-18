import cv2
import tensorflow as tf
import numpy as np

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
