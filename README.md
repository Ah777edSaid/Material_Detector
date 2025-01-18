
# MobileNet Material Detection

This project uses a pre-trained **MobileNetV2** model for detecting materials in images. It leverages **TensorFlow**, **OpenCV**, and **NumPy** to process images and detect objects with their corresponding labels and confidence scores.

## Requirements

To run the project, you need the following Python libraries:

- `opencv-python` – For image processing.
- `tensorflow` – For loading and using the pre-trained MobileNetV2 model.
- `numpy` – For handling numerical operations on images.

You can install these dependencies by running the following command:

```bash
pip install opencv-python tensorflow numpy
```

## Output

The output will show predictions for materials in the provided images. The predictions are returned as the class name with the associated confidence score. You can view the ImageNet class labels for various materials and objects by visiting this URL:

- [ImageNet Class Index](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json)

## How It Works

1. The image is preprocessed (resized and formatted) for compatibility with the MobileNetV2 model.
2. The model predicts the class of objects in the image.
3. The results, including class labels and confidence scores, are printed and can be visualized in the output image.

### Example Output:
```text
Predicted material: "wood" with confidence 0.89
```
