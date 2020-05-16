import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence
from skimage.transform import resize

from keras_yolo.model import Yolo


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


aug_pipe = iaa.Sequential(
    [iaa.SomeOf(
        (0, 1),
        [iaa.Invert(0.05, per_channel=True)],
        random_order=True
    )],
    random_order=True,
)


class BatchGenerator(Sequence):
    def __init__(
        self,
        images,
        batch_size,
        labels,
        n_anchors,
        max_train_boxes=10,
        shuffle=True,
        augment=True,
    ):
        self.images = images
        self.shuffle = shuffle
        self.augment = augment
        if self.shuffle:
            np.random.shuffle(self.images)

        self.batch_size = batch_size
        self.labels = labels
        self.n_anchors = n_anchors
        self.max_train_boxes = max_train_boxes

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def load_image_and_prepare_boxes(self, filename, height, width, object):
        grid_h, grid_w = Yolo.GRID_SIZE
        n_classes = len(self.labels)

        # Load the image and normalize it in a very simple way.
        image = resize(cv2.imread(filename), Yolo.INPUT_SIZE, mode='constant')

        if self.augment:
            image = aug_pipe.augment_image(image)

        boxes = np.zeros((self.max_train_boxes, 4), 'float32')
        classes = np.zeros((self.max_train_boxes, n_classes))

        for i, obj in enumerate(object):
            if i >= self.max_train_boxes:
                # We cannot store more than MAX_TRAIN_BOXES so simply skip those
                break
            if (
                ('xmax' not in obj)
                or ('xmin' not in obj)
                or ('ymax' not in obj)
                or ('ymin' not in obj)
            ):
                # If polygon we cannot parse for now
                print(filename)

            # Standardize all encodings to 0, 1 so we can easily use them
            # in subsequent calculations.
            w = (obj['xmax'] - obj['xmin']) / width
            h = (obj['ymax'] - obj['ymin']) / height
            x_std = (obj['xmin'] / width) + 0.5 * w
            y_std = (obj['ymin'] / height) + 0.5 * h

            # Next we convert to predictions that match up with Yolo's grid
            # this means we scale them with the grid size

            w = w * grid_w
            h = h * grid_h
            x_std = x_std * grid_w
            y_std = y_std * grid_h

            boxes[i] = np.array([x_std, y_std, w, h])

            # Fill the one hot encoding of the classes indicator
            class_index = self.labels.get(obj['name'])
            classes[i, class_index] = 1

        return image, boxes, classes

    def __getitem__(self, idx):
        input_h, input_w = Yolo.INPUT_SIZE
        grid_h, grid_w = Yolo.GRID_SIZE

        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.batch_size

        img_input = np.zeros(
            (self.batch_size, input_h, input_w, 3),
            'float32'
        )
        boxes_input = np.zeros(
            (self.batch_size, self.max_train_boxes, 4),
            'float32'
        )
        conf_input = np.zeros(
            (self.batch_size, self.max_train_boxes, 1),
            'float32'
        )
        classes_input = np.zeros(
            (self.batch_size, self.max_train_boxes, len(self.labels)), 'float32'
        )

        for i, train_instance in enumerate(self.images[l_bound:r_bound]):
            image, boxes, classes = (
                self.load_image_and_prepare_boxes(**train_instance)
            )

            img_input[i] = image
            boxes_input[i] = boxes
            classes_input[i] = classes

        y = np.concatenate([boxes_input, conf_input, classes_input], axis=-1)
        y = np.tile(
            y[:, None, None, None, ...],
            [1, grid_h, grid_w, self.n_anchors, 1, 1]
        )

        return img_input[..., [2, 1, 0]], y

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.batch_size))
