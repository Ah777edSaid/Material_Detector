from setuptools import setup, find_packages

setup(
    name='image_classifier',  # اسم الحزمة
    version='0.1',  # النسخة الحالية
    packages=find_packages(),  # العثور على جميع الحزم داخل المجلد
    install_requires=[  # الحزم المطلوبة
        'tensorflow>=2.0',  # TensorFlow ضروري لتدريب النموذج
        'opencv-python',  # OpenCV لمعالجة الصور
        'numpy',  # Numpy للتعامل مع المصفوفات
    ],
    description='An image classifier using MobileNetV2 for material prediction',  # وصف الحزمة
    author='Ah777ed Said',  # اسم المؤلف
    author_email='ahmed01226313482@gmail.com',  # البريد الإلكتروني للمؤلف
    url='https://github.com/Ah777edSaid/MobileNet',  # رابط الريبو في GitHub
)
