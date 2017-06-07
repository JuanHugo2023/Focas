from keras_density import *

model = build_model()
model.load_weights('keras_density.hdf5')

predict_test_images(model, 'test.csv')
