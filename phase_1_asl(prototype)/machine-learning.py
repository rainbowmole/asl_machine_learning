import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_SIZE = 96       # MobileNetV2 needs at least 96x96
BATCH_SIZE = 32
EPOCHS_FROZEN = 10  # train only the head first
EPOCHS_FINETUNE = 20  # then fine-tune

# --- Data generators ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    horizontal_flip=False
)

train_data = datagen.flow_from_directory(
    'asl_alphabet_train/asl_alphabet_train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    'asl_alphabet_train/asl_alphabet_train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

NUM_CLASSES = train_data.num_classes  # 29

# --- Compute class weights to handle any imbalance ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))

# --- Transfer learning with MobileNetV2 ---
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base initially

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_loss'),
    ModelCheckpoint('asl_model_best.keras', save_best_only=True, monitor='val_accuracy')
]

# --- Phase 1: train head only (base frozen) ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("=== Phase 1: Training head (base frozen) ===")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# --- Phase 2: unfreeze top layers and fine-tune ---
base_model.trainable = True
for layer in base_model.layers[:-30]:  # keep early layers frozen
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("=== Phase 2: Fine-tuning top layers ===")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

model.save('asl_model.h5')
print("Model saved to asl_model.h5")
