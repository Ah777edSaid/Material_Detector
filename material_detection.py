import cv2
import tensorflow as tf
import numpy as np
import requests


class ImageDownloader:
    def __init__(self, image_url, save_path):
        self.image_url = image_url
        self.save_path = save_path

    def download(self):
        response = requests.get(self.image_url)
        if response.status_code == 200:  # Check if the request was successful
            with open(self.save_path, 'wb') as file:
                file.write(response.content)
            print(f"Image downloaded successfully and saved to {self.save_path}")
        else:
            print("Failed to retrieve the image")


class ImagePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def preprocess(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (224, 224))  # Resize to 224x224 to fit MobileNet
        image = np.expand_dims(image, axis=0)  # Add extra dimension for batch size (1, 224, 224, 3)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess the image for MobileNet
        return image


class MaterialPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, image_path):
        preprocessor = ImagePreprocessor(image_path)
        processed_image = preprocessor.preprocess()
        predictions = self.model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
        print(f"Predicted material: {decoded_predictions[1]} with confidence {decoded_predictions[2]:.2f}")
        return decoded_predictions


class MobileNetModel:
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(weights='imagenet')

    def get_model(self):
        return self.model


# Main execution
if __name__ == "__main__":
    # Download image
    image_url = "https://th.bing.com/th/id/OIP.VCZorWbps0gxQVPO9yOJawHaFj?rs=1&pid=ImgDetMain"
    save_path = "preprocess_image.jpg"
    downloader = ImageDownloader(image_url, save_path)
    downloader.download()

    # Load the MobileNet model
    mobilenet = MobileNetModel()
    model = mobilenet.get_model()

    # Predict material type
    predictor = MaterialPredictor(model)
    image_path = "preprocess_image.jpg"
    predictor.predict(image_path)
