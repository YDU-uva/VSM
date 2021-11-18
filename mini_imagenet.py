import os
import sys
import numpy as np
import pickle


def onehottify_2d_array(a):
    """
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    :param a: 2-dimensional array.
    :return: 3-dim array where last dim corresponds to one-hot encoded vectors.
    """

    # https://stackoverflow.com/a/46103129/ @Divakar
    def all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    num_columns = a.max() + 1
    out = np.zeros(a.shape + (num_columns,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


class MiniImageNetData(object):

    def __init__(self, path, seed):
        """
        Constructs a miniImageNet dataset for use in episodic training.
        :param path: Path to miniImageNet data files.
        :param seed: Random seed to reproduce batches.
        """
        np.random.seed(seed)

        self.image_height = 84
        self.image_width = 84
        self.image_channels = 3

        path_train = os.path.join(path, 'mini_imagenet_train.pkl')
        path_validation = os.path.join(path, 'mini_imagenet_val.pkl')
        path_test = os.path.join(path, 'mini_imagenet_test.pkl')
        # label_path_train = os.path.join(path, 'label_mini_imagenet_train.pkl')
        # label_path_validation = os.path.join(path, 'label_mini_imagenet_val.pkl')
        # label_path_test = os.path.join(path, 'label_mini_imagenet_test.pkl')

        self.train_set = pickle.load(open(path_train, 'rb'))
        self.validation_set = pickle.load(open(path_validation, 'rb'))
        self.test_set = pickle.load(open(path_test, 'rb'))

        # self.train_set_label = pickle.load(open(label_path_train, 'rb'))
        # self.validation_set_label = pickle.load(open(label_path_validation, 'rb'))
        # self.test_set_label = pickle.load(open(label_path_test, 'rb'))

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    def _sample_batch(self, images, tasks_per_batch, shot, way, eval_samples, source):
        """
        Sample a k-shot batch from images.
        :param images: Data to sample from [way, samples, h, w, c] (either of train, val, test)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: A list [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * (shot or eval_samples), h, w, c]
            * Labels: [tasks_per_batch, way * (shot or eval_samples), way]
                      (one-hot encoded in last dim)
        """

        samples_per_class = shot + eval_samples

        # Set up empty arrays
        train_images = np.empty((tasks_per_batch, way, shot, self.image_height, self.image_width,
                                 self.image_channels), dtype=np.float32)
        test_images = np.empty((tasks_per_batch, way, eval_samples, self.image_height, self.image_width,
                                self.image_channels), dtype=np.float32)
        ori_train_labels = np.empty((tasks_per_batch, way, shot), dtype=np.int32)
        ori_test_labels = np.empty((tasks_per_batch, way, eval_samples), dtype=np.int32)

        train_labels = np.empty((tasks_per_batch, way, shot), dtype=np.int32)
        test_labels = np.empty((tasks_per_batch, way, eval_samples), dtype=np.int32)


        classes_idx = np.arange(images.shape[0])


        samples_idx = np.arange(images.shape[1])

        exist_classes = np.empty((tasks_per_batch, way), dtype=np.int32)
        # fill arrays one task at a time
        for i in range(tasks_per_batch):
            choose_classes = np.random.choice(classes_idx, size=way, replace=False)
            if source == 'train':
                exist_classes[i] = choose_classes
            elif source == 'validation':
                exist_classes[i] = choose_classes + 64
            elif source == 'test':
                exist_classes[i] = choose_classes + 80

            shape_imgs = images[choose_classes, :samples_per_class].shape
            imgs_tmp = np.zeros(shape_imgs)
            ori_lab_tmp = np.zeros((shape_imgs[0], shape_imgs[1]))
            for j in range(way):
                choose_samples = np.random.choice(samples_idx, size=samples_per_class, replace=False)
                imgs_tmp[j, ...] = images[choose_classes[j], choose_samples, ...]
                if source == 'train':
                    ori_lab_tmp[j] = np.zeros(samples_per_class) + choose_classes[j]
                elif source == 'validation':
                    ori_lab_tmp[j] = np.zeros(samples_per_class) + choose_classes[j] + 64
                elif source == 'test':
                    ori_lab_tmp[j] = np.zeros(samples_per_class) + choose_classes[j] + 80

                ori_lab_tmp[j] = np.zeros(samples_per_class) + choose_classes[j]
            labels_tmp = np.arange(way)

            train_images[i] = imgs_tmp[:, :shot].astype(dtype=np.float32)
            test_images[i] = imgs_tmp[:, shot:].astype(dtype=np.float32)

            ori_train_labels[i] = ori_lab_tmp[:, :shot]
            ori_test_labels[i] = ori_lab_tmp[:, shot:]

            train_labels[i] = np.expand_dims(labels_tmp, axis=1)
            test_labels[i] = np.expand_dims(labels_tmp, axis=1)

        # reshape arrays
        train_images = train_images.reshape(
            (tasks_per_batch, way * shot, self.image_height, self.image_width, self.image_channels)) / 255.
        test_images = test_images.reshape(
            (tasks_per_batch, way * eval_samples, self.image_height, self.image_width, self.image_channels)) / 255.
        ori_train_labels = ori_train_labels.reshape((tasks_per_batch, way * shot))
        ori_test_labels = ori_test_labels.reshape((tasks_per_batch, way * eval_samples))
        train_labels = train_labels.reshape((tasks_per_batch, way * shot))
        test_labels = test_labels.reshape((tasks_per_batch, way * eval_samples))

        # labels to one-hot encoding
        train_labels = onehottify_2d_array(train_labels)
        test_labels = onehottify_2d_array(test_labels)

        return [train_images, test_images, train_labels, test_labels, ori_train_labels, ori_test_labels, exist_classes]

    def _shuffle_batch(self, train_images, train_labels, ori_train_labels):
        """
        Randomly permute the order of the second column
        :param train_images: [tasks_per_batch, way * shot, height, width, channels]
        :param train_labels: [tasks_per_batch, way * shot, way]
        :return: permuted images and labels.
        """
        for i in range(train_images.shape[0]):
            permutation = np.random.permutation(train_images.shape[1])
            train_images[i, ...] = train_images[i, permutation, ...]
            ori_train_labels[i] = ori_train_labels[i, permutation]
            train_labels[i, ...] = train_labels[i, permutation, ...]
        return train_images, train_labels, ori_train_labels

    def get_batch(self, source, tasks_per_batch, shot, way, eval_samples):
        """
        Returns a batch of tasks from miniImageNet. Values are np.float32 and scaled to [0,1]
        :param source: one of `train`, `test`, `validation` (i.e. from which classes to pick)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * shot, height, width, channels]
            * Labels: [tasks_per_batch, way * shot, way]
                      (one-hot encoded in last dim)
        """

        # sample a batch
        if source == 'train':
            images = self.train_set
            # labels = self.train_set_label
        elif source == 'validation':
            images = self.validation_set
            # labels = self.validation_set_label
        elif source == 'test':
            images = self.test_set
            # labels = self.test_set_label

        train_images, test_images, train_labels, test_labels, ori_train_labels, ori_test_labels, exist_classes = self._sample_batch(
            images, tasks_per_batch, shot, way,
            eval_samples, source)

        train_images, train_labels, ori_train_labels = self._shuffle_batch(train_images, train_labels, ori_train_labels)

        return [train_images, test_images, train_labels, test_labels, ori_train_labels, ori_test_labels, exist_classes]

    def _sample_memory_batch(self, images, tasks_all, eval_samples):

        samples_per_class = eval_samples

        classes_idx = np.arange(images.shape[0])
        samples_idx = np.arange(images.shape[1])

        # fill arrays one task at a time


        shape_imgs = images[classes_idx, :samples_per_class].shape
        imgs_tmp = np.zeros(shape_imgs)
        ori_lab_tmp = np.zeros((shape_imgs[0], shape_imgs[1]))

        for i in range(tasks_all):
            choose_samples = np.random.choice(samples_idx, size=samples_per_class, replace=False)
            imgs_tmp[i, ...] = images[classes_idx[i], choose_samples, ...]
            ori_lab_tmp[i] = np.zeros(samples_per_class) + classes_idx[i]
        train_images = imgs_tmp.astype(dtype=np.float32)
        ori_train_labels = ori_lab_tmp.astype(dtype=np.int32)

        # reshape arrays
        train_images = train_images.reshape(
            (tasks_all, eval_samples, self.image_height, self.image_width, self.image_channels)) / 255.

        return [train_images, ori_train_labels]

    def get_memory_batch(self, tasks_all, eval_samples):
        """
        Returns a batch of tasks from miniImageNet. Values are np.float32 and scaled to [0,1]
        :param source: one of `train`, `test`, `validation` (i.e. from which classes to pick)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * shot, height, width, channels]
            * Labels: [tasks_per_batch, way * shot, way]
                      (one-hot encoded in last dim)
        """

        images = self.train_set

        train_images, ori_train_labels = self._sample_memory_batch(images, tasks_all, eval_samples)

        return [train_images, ori_train_labels]
