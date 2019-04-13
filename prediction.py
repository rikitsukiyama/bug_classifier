import json
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

#model = ResNet50(weights='imagenet')
model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')

graph = tf.get_default_graph()
CLASS_INDEX = json.load(open('static/files/imagenet_class_index.json'))

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
        return self.decode_predictions(preds, top=3)[0]

    def ioPredict(self, image):
        image = Image.open(io.BytesIO(image))
        # preprocess the image and prepare it for classification
        image = self.prepare_image(image, target=(224, 224))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = self.model.predict(image)
        return preds

    def decode_predictions(self, preds, top=5):
        if len(preds.shape) != 2 or preds.shape[1] != 1000:
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 1000)). '
                             'Found array with shape: ' + str(preds.shape))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
            results.append(result)
        return results
