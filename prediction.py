import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

model = ResNet50(weights='imagenet')
graph = tf.get_default_graph()

class classifier(object):
    """
    Pretrained net: https://keras.io/applications/#usage-examples-for-image-classification-models
    """
    def __init__(self):
        self.model = model

    def prepare_image(self, image, target):
        # if the image mode is not RGB, convert it
        if image.mode != "RGB":
            image = image.convert("RGB")

        # resize the input image and preprocess it
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # return the processed image
        return image

    def predict(self, image_path):
        img = image.load_img(image_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        with graph.as_default():
            preds = self.model.predict(x)
        return decode_predictions(preds, top=3)[0]

    def ioPredict(self, image):
        image = Image.open(io.BytesIO(image))
        # preprocess the image and prepare it for classification
        image = self.prepare_image(image, target=(224, 224))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = self.model.predict(image)
        return preds
