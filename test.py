import os
for dir in ['test','train','valid']:
  os.makedirs(f'data/{dir}',exist_ok=True)

import numpy as np
import cv2


dir = 'data/valid'
img_dir = f"{dir}/images"
mask_dir = f"{dir}/masks"
img_size = (224, 224)  # taille d'entrée du modèle
files = [file for file in os.listdir(img_dir) if file.endswith('.jpg')]
images = [cv2.imread(os.path.join(img_dir,file)) for file in files]
masks = [cv2.imread(os.path.join(mask_dir,file.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE) for file in files]

X, Y = [], []

for img_id,(img,mask) in enumerate(zip(images,masks)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    X.append(img_resized / 255.0)  # normaliser image
    Y.append(mask)         # masque binaire

# Conversion en numpy
X_val = np.array(X, dtype=np.float32)
Y_val = np.array(Y, dtype=np.uint8)
Y_val = np.expand_dims(Y_val, axis=-1)  # (H, W, 1)




dir = 'data/train'
img_dir = f"{dir}/images"
mask_dir = f"{dir}/masks"
img_size = (224, 224)  # taille d'entrée du modèle
files = [file for file in os.listdir(img_dir) if file.endswith('.jpg')]
images = [cv2.imread(os.path.join(img_dir,file)) for file in files]
masks = [cv2.imread(os.path.join(mask_dir,file.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE) for file in files]

X, Y = [], []

for img_id,(img,mask) in enumerate(zip(images,masks)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    X.append(img_resized / 255.0)  # normaliser image
    Y.append(mask)         # masque binaire
# Conversion en numpy
X_train = np.array(X, dtype=np.float32)
Y_train = np.array(Y, dtype=np.uint8)
Y_train = np.expand_dims(Y_train, axis=-1)  # (H, W, 1)



import tensorflow as tf
from tensorflow.keras import layers as L, models as M
from tensorflow.keras.applications import EfficientNetB7

# --------- Blocks ----------
def conv_block(x, filters, kernel_size=3, bn=True):
    x = L.Conv2D(filters, kernel_size, padding="same")(x)
    if bn: x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Conv2D(filters, kernel_size, padding="same")(x)
    if bn: x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    return x

def resize_to(skip, x):
    """resize skip tensor to match x spatial dims"""
    return L.Resizing(x.shape[1], x.shape[2], interpolation="bilinear")(skip)

def up_block(x, skip, filters):
    x = L.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    if skip is not None:
        skip = L.Conv2D(filters, 1, padding="same")(skip)
        skip = resize_to(skip, x)
        x = L.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

# --------- Model ----------
def build_unet_efficientnetb7(input_shape=(224,224,3), freeze_encoder=True):
    base = EfficientNetB7(include_top=False, weights="imagenet", input_shape=input_shape)

    # Skip connections (repères connus dans EfficientNetB7)
    skips = [
        base.get_layer("block2a_expand_activation").output,  # ~56x56
        base.get_layer("block3a_expand_activation").output,  # ~28x28
        base.get_layer("block4a_expand_activation").output,  # ~14x14
        base.get_layer("block6a_expand_activation").output,  # ~7x7
    ]
    bottleneck = base.get_layer("top_activation").output  # ~7x7

    if freeze_encoder:
        for l in base.layers:
            l.trainable = False

    # Decoder
    x = bottleneck
    x = up_block(x, skips[-1], 512)  # 7 -> 14
    x = up_block(x, skips[-2], 256)  # 14 -> 28
    x = up_block(x, skips[-3], 128)  # 28 -> 56
    x = up_block(x, skips[-4], 64)   # 56 -> 112
    x = up_block(x, None, 32)        # 112 -> 224

    outputs = L.Conv2D(1, 1, activation="sigmoid", padding="same")(x)

    # 🔑 corrige la sortie pour matcher l’input
    outputs = L.Resizing(input_shape[0], input_shape[1])(outputs)

    model = M.Model(inputs=base.input, outputs=outputs, name="UNet_EfficientNetB7")
    return model












model = build_unet_efficientnetb7(input_shape=(224,224,3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy","recall","precision"]
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=16
)
model.save("unet_efficientnetb7.keras")