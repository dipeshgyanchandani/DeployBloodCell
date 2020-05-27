
#::: Import modules and packages :::
# Flask utils



from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import optimizers

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = './model/model_new_30adam3030_20191030_01.json'
MODEL_WEIGHTS = './model/model_new_30_eopchs_adam_20191030_01.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(150, 150))
    img_arr = np.expand_dims(img_to_array(img), axis=0)
    datagen = ImageDataGenerator(rescale=1./255)
    for batch in datagen.flow(img_arr, batch_size=1):
     img = batch
     break
    
    model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc'])
    preds = model.predict_classes(img)
    
    return preds


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = {'TRAIN': ['NORMAL', 'INFECTED'],
               'TEST': ['NORMAL', 'INFECTED']}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)

		predicted_class = prediction[0]
		print('We think that is {}.'.format(predicted_class))

		return str(predicted_class).lower()

if __name__ == '__main__':
	app.run(debug = True)
