
from keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.h5')
model.make_predict_function()
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = model.predict(i)
	if(p[0][0] > p[0][1]):
		return "Mask on"
	else:
		return "No Mask detected"
