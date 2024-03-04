import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings



warnings.filterwarnings("ignore")

imgs = glob.glob("/home/cemalgursel/Derin+Öğrenme+ile+Görüntü+İşleme/TREX_GAME_WİTH_PYTHON/img_nihai/*.png")
witdh = 125
height = 50

X = []
Y = []

for img in imgs:
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((witdh,height)))
    im = im/255
    X.append(im)
    Y.append(label)

X = np.array(X)
X = X.reshape(X.shape[0],witdh,height)

def onehot_label(values):
    label_encoder = LabelEncoder()
    integers_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integers_encoded = integers_encoded.reshape(len(integers_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integers_encoded)
    return onehot_encoded

Y = onehot_label(Y)

train_X, test_X, train_y, test_y = train_test_split(X,Y,test_size=0.25,random_state=2)

#cnn Model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(witdh,height,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

model.fit(train_X,train_y,epochs=35,batch_size=64)

score_train = model.evaluate(train_X,train_y)
print("Egitim dogrulugu: %",score_train[1]*100)

score_test = model.evaluate(test_X,test_y)
print("Test dogrulugu: %",score_test[1]*100)

open("model.json","w").write(model.to_json())
model.save_weights("trex.weights.h5")
