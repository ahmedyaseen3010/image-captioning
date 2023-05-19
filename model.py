from tensorflow.keras.models import load_model
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

model = VGG16()

model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

features = {}
directory ="D:\\Lec's\\caption\\Flicker8k_Dataset"

with open('features.pkl', 'rb') as f:features = pickle.load(f)

with open("D:\\Lec's\\caption\\captions.txt", 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = {}

for line in captions_doc.split('\n'):
    tokens = line.split(',')
    
    if len(line) < 2:
        continue
        
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    
    if image_id not in mapping:
        mapping[image_id] = []

    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):

            caption = captions[i]

            caption = caption.lower()

            caption = caption.replace('[^A-Za-z]', '')

            caption = caption.replace('\s+', ' ')

            caption = 'start  ' + " ".join([word for word in caption.split() if len(word)>1]) + '  end'
            captions[i] = caption

clean(mapping)
all_captions = []
for key in mapping:
	for caption in mapping[key]:
		all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
      
	X1, X2, y = list(), list(), list()
	n = 0
	while 1:
		for key in data_keys:
			n += 1
			captions = mapping[key]
			for caption in captions:
				seq = tokenizer.texts_to_sequences([caption])[0]
				for i in range(1, len(seq)):
					in_seq, out_seq = seq[:i], seq[i]
					in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
					out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
					X1.append(features[key][0])
					X2.append(in_seq)
					y.append(out_seq)
			if n == batch_size:
				X1, X2, y = np.array(X1), np.array(X2), np.array(y)
				yield [X1, X2], y
				X1, X2, y = list(), list(), list()
				n = 0

# encoder model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)


# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
 
# plot the model
plot_model(model, show_shapes=True)


# Load the model
model = load_model("C:\\neural\\best_model.h5")

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
        	return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'start'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        text = model.predict([image, sequence], verbose=0)
        text = np.argmax(text)
        word = idx_to_word(text, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'end':
            break

    return in_text

from PIL import Image
import matplotlib.pyplot as plt

def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = "D:\\caption\\Flicker8k_Dataset\\"+image_name
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)

# generate_caption("10815824_2997e03d76.jpg")

vgg_model = VGG16() 
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

image_path = "C:\\projects\\pic.jpg"

image1 = load_img(image_path, target_size=(224, 224))
image = img_to_array(image1)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

feature = vgg_model.predict(image, verbose=0)

x=predict_caption(model, feature, tokenizer, max_length)

plt.imshow(image1)
print(x)