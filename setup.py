from setuptools import setup

setup(
    name='Material_Detector',  # اسم الحزمة
    version='0.1',  # النسخة
    py_modules=['Detector'],  # تحديد الملف الذي يحتوي على الكود
    install_requires=[  # الحزم المطلوبة
        'tensorflow>=2.0',
        'opencv-python',
        'numpy',
    ],
    description='An image classifier using MobileNetV2 for material prediction',
    author='Ah777ed Said',
    author_email='ahmed01226313482@gmail.com',  # البريد الإلكتروني للمؤلف
    url='https://github.com/Ah777edSaid/MobileNet',  # رابط الريبو
)
