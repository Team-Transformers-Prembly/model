#model prediction pipeline
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Load model
model = tf.keras.models.load_model("model.h5")

class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight','Tomato_Late_blight', 'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self,img_object):
        return self
    
    def transform(self,img_object):
        img_array = image.img_to_array(img_object)
        expanded = (np.expand_dims(img_array,axis=0))
        return expanded

class Predictor(BaseEstimator, TransformerMixin):
    def fit(self,img_array):
        return self
    
    def predict(self,img_array):
        probabilities = model.predict(img_array)
        predicted_class = class_names[np.argmax(probabilities)]
        return predicted_class

full_pipeline = Pipeline([('preprocessor',Preprocessor()),
                         ('predictor',Predictor())])

prediction = full_pipeline.predict(image) #image is the uploaded file
print('Health status/Disease:\n{}'.format(class_names[np.argmax(prediction)])) 
