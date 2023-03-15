import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
from tensorflow.train import Checkpoint
import tensorflow.image as image
import tensorflow.data as data

train = utils.image_dataset_from_directory(
    'balls/train',
    label_mode = "categorical",
    # color_mode = 'grayscale',
    batch_size = 32,
    image_size = (224, 224),
    # seed = 8008,
    # validation_split = 0.3,
    # subset = "training",
)

test = utils.image_dataset_from_directory(
    'balls/valid',
    label_mode = "categorical",
    # color_mode = 'grayscale',
    batch_size = 32,
    image_size = (224, 224),
    # seed = 8008,
    # validation_split = 0.3,
    # subset = "validation",
)

print("Class Names:")
print(train.class_names)
class_names = train.class_names

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = data.AUTOTUNE)

train_brightness = train.map(lambda x, y: (image.stateless_random_brightness(x, 0.5, (2,3)), y))
train_contrast = train.map(lambda x, y: (image.stateless_random_contrast(x, 0.2, 0.5, (2,3)), y))
train_flip_left = train.map(lambda x, y: (image.stateless_random_flip_left_right(x, (2,3)), y))
train_flip_up = train.map(lambda x, y: (image.stateless_random_flip_up_down(x, (2,3)), y))
train_hue = train.map(lambda x, y: (image.stateless_random_hue(x, 0.3, (2,3)), y))
train_saturation = train.map(lambda x, y: (image.stateless_random_saturation(x, 0.25, 1, (2,3)), y))

train = train.concatenate(train_contrast)
train = train.concatenate(train_brightness)
train = train.concatenate(train_flip_left)
train = train.concatenate(train_flip_up)
train = train.concatenate(train_hue)
train = train.concatenate(train_saturation)

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        """
        self.model.add(layers.RandomContrast(factor = 0.5))
        self.model.add(layers.RandomRotation(factor = 0.5))
        self.model.add(layers.RandomZoom(height_factor = 0.5))
        """
        self.model.add(layers.Conv2D(
            8, # filters
            6, # kernels
            strides = 2, # a.k.a step size
            activation = "relu", 
            input_shape = input_shape,
        )) # output: 110 x 110 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        )) # output: 55 x 55 x 8
        self.model.add(layers.Conv2D(
            8, # filters
            2, # kernels
            strides = 1, # a.k.a step size
            activation = "relu", 
        )) # output: 54 x 54 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        )) # output: 27 x 27 x 8
        self.model.add(layers.Conv2D(
            8, # filters
            2, # kernels
            strides = 1, # a.k.a step size
            activation = "relu", 
        )) # output: 26 x 26 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        )) # output: 13 x 13 x 8
        self.model.add(layers.Flatten(
        ))# output: 1352
        self.model.add(layers.Dense(
            1024,
            activation = "relu"
        ))
        self.model.add(layers.Dense(
            512,
            activation = "relu"
        ))
        self.model.add(layers.Dense(
            256,
            activation = "relu"
        ))
        self.model.add(layers.Dense(
            128,
            activation = "relu"
        ))
        self.model.add(layers.Dense(
            30, # Exactly equal to number of classes
            activation = "softmax", # Always use softmax on your last layer
        ))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy']
        )
    def __str__(self):
        self.model.summary()
        return ""
    def save(self, filename):
        self.model.save(filename)

net = Net((224, 224, 3))
checkpoint = Checkpoint(net.model)

callbacks = [
    callbacks.ModelCheckpoint(
        'checkpoints/checkpoints_{epoch:03d}', 
        verbose = 2, 
        save_freq = 791,
    )
]

print(net)
net.model.fit(
    train,
    batch_size = 32,
    epochs = 200,
    verbose = 1,
    validation_data = test,
    validation_batch_size = 32,
    callbacks = callbacks,
)

save_path = "saves/balls_model_save"
net.model.save(save_path)
with open(f'{save_path}/class_names.data', 'wb') as f:
    pickle.dump(class_names, f)