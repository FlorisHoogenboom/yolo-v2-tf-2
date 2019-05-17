
import numpy as np
import tensorflow as tf

from keras_yolo.model import Yolo
from keras_yolo.preprocessing import parse_annotation, BatchGenerator
from keras_yolo.utils import WeightReader


# In[5]:


ANCHORS = [
    [1, 1],
    [1, 1.5],
    [1.5, 1],
    [3, 3],
    [1.5, 4],
    [4, 1.5]
]

MAX_TRAIN_BOXES = 10


# # Load the training data
# We parse the PASCAL VOC definitions of the images to a format we can handle. Next, we define a generator to prepare input and output data for our network.

# In[6]:


image_path = '/content/data/images/'
annot_path = '/content/data/annotations/'

all_imgs, seen_labels = parse_annotation(annot_path, image_path)

labels = list(seen_labels.keys())
label_mapping = dict((label, index) for index, label in enumerate(labels))

print("We have seen the following labels (format: label, counts): {}".format(seen_labels))


# Define the generator used for training
bg = BatchGenerator(all_imgs, 5, label_mapping, len(ANCHORS), MAX_TRAIN_BOXES)


# # Define the model and load pretrained weights.

# In[7]:


yolo = Yolo(len(labels), anchors=ANCHORS, max_boxes=MAX_TRAIN_BOXES)


# We will set the model using the pretrained weights downloaded form the YOLO website.

# In[8]:


wt_path = '/content/weights/yolo.weights'
weight_reader = WeightReader(wt_path)

weight_reader.reset()
nb_conv = 22

for i in range(1, nb_conv+1):
    conv_layer = yolo.get_layer('conv_' + str(i))

    if i < nb_conv:
        norm_layer = yolo.get_layer('norm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])


# ### Test the basic structure of the model
# To test the basic structure of the model we make a few dummy predictions. This should give us an idea whether all values are properly normalized and whether the model graph is correctly constructed.

# # Warm up training loss
# Let's investigate how to compute the base loss. That is: how to compute a loss between the outpus and the base anchors. This is needeed for warmup training.

# In[9]:


from keras.optimizers import Adam
# optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# #
# yolo.compile(optimizer, yolo.warmup_loss)
#
#
# # In[10]:
#
#
# yolo.fit_generator(generator        = bg,
#                    steps_per_epoch  = 10,
#                    epochs           = 1,
#                    verbose          = 1,
#                    max_queue_size   = 3)
#
#
# # In[11]:


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
yolo.compile(optimizer, yolo.loss)


# In[12]:


yolo.fit_generator(generator        = bg,
                   steps_per_epoch  = len(all_imgs)/5,
                   epochs           = 30,
                   verbose          = 1,
                   max_queue_size   = 3)


