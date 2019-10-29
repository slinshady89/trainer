import cv2
import os
import numpy as np
from model import unet
from segnet import segnet
from model_wIndices import unet_wIndices
from keras.preprocessing.image import img_to_array
from keras.utils import multi_gpu_model, plot_model, print_summary
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend as K


class Trainer(object):
    def __init__(self, _train_list, _val_list, _inf_list, _dag_it = 0, _input_shape = (256, 1024, 3),
                 _train_steps = 500, _val_steps = 200, _num_epochs = 15, _batch_size = 4, _gpu_num = '0, 1',
                 _no_inidices = True, _segnet = False):
        self.dag_it = _dag_it
        self.train_list = _train_list
        self.val_list = _val_list
        self.inf_list = _inf_list
        self.base_dir = '/media/localadmin/Test/11Nils/kitti/dataset/sequences/Data/'
        self.img_dir = 'images/'
        self.label_dir = 'labels/'
        self.inf_dir = 'inf/'
        self.dag_dir = 'dagger/'
        self.log_dir = 'log/'
        self.optimizer = 'adagrad'
        self.gpu_num = _gpu_num  # '1'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_num
        self.untrained = 'store_true'
        self.loss = 'categorical_crossentropy'
        self.output_mode = 'softmax'
        self.pool_size = (2, 2)
        self.kernel = 3
        self.input_shape = _input_shape  # (128, 512, 3)
        self.n_labels = 3  # num classes
        self.val_steps = _val_steps
        self.epoch_steps = _train_steps
        self.n_epochs = _num_epochs
        self.batch_size = _batch_size
        self.filters = 8
        self.b_pool_indices = _no_inidices
        self.b_use_segnet = _segnet
        if not self.b_pool_indices and not self.b_use_segnet:
            self.model = unet_wIndices(self.input_shape, self.n_labels, self.filters, self.kernel, self.pool_size,
                                       self.output_mode)
        elif not self.b_use_segnet:
            self.model = unet(self.input_shape, self.n_labels, self.filters, self.kernel, self.pool_size,
                              self.output_mode)
        else:
            self.model = segnet(self.input_shape, self.n_labels, self.filters, self.kernel, self.pool_size,
                                self.output_mode)
        print(self.model.summary())
        list_gpus_trained = [int(x) for x in self.gpu_num.split(',')]
        self.num_gpus = len(list_gpus_trained)
        if self.num_gpus > 1:
            trained_gpu_str = ', '.join(str(e) for e in list_gpus_trained)
            print('Training on GPU\'s: ' + trained_gpu_str)
            self.multi_model = multi_gpu_model(self.model, gpus = self.num_gpus)
        else:
            self.multi_model = self.model
        self.multi_model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])
        plot_model(model = self.multi_model, to_file = self.base_dir + 'model.png')
        print(print_summary(self.multi_model))
        self.std = [0.32636853, 0.31895106, 0.30716496]
        self.mean = [0.39061851, 0.38151629, 0.3547171]
        self.es_cb = []
        self.tb_cb = []
        self.cp_cb = []

    # generator that we will use to read the data from the directory
    def data_gen(self, lists):
        # mean color values and their standard deviation of KITTI data
        while True:
            ix = np.random.choice(np.arange(len(lists)), self.batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                image_name = lists[i]  # os.path.join("%06d" % i)
                original_img = cv2.imread(self.base_dir + self.img_dir + image_name)
                # masks
                original_mask = cv2.imread(self.base_dir + self.label_dir + image_name)
                if original_mask is None:
                    print(self.base_dir + self.label_dir + image_name)
                array_img = self.crop_resize_norm_bgr(original_img, self.input_shape)
                array_mask = self.crop_resive_mask(original_mask, self.input_shape)
                imgs.append(array_img)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels

    def crop_resize_norm_bgr(self, img, dims):
        h, w, c = img.shape
        cropped_img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_img = cv2.resize(cropped_img, (dims[1], dims[0]))
        normed = resized_img / 255.0
        mean_free = (normed[:, :] - self.mean) / self.std
        array_img = img_to_array(mean_free)
        return array_img

    def crop(self, img, dims):
        h, w, c = img.shape
        cropped_img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_img = cv2.resize(cropped_img, (dims[1], dims[0]))
        return resized_img / 255.0

    def crop_resive_mask(self, mask, dims):
        h, w, c = mask.shape
        cropped_mask = mask[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_mask = cv2.resize(cropped_mask, (dims[1], dims[0]))
        array_mask = img_to_array(resized_mask) / 255
        return array_mask

    def train(self):

        history = self.multi_model.fit_generator(generator = self.data_gen(self.train_list),
                                                 steps_per_epoch = self.epoch_steps,
                                                 epochs = self.n_epochs,
                                                 validation_data = self.data_gen(self.val_list),
                                                 validation_steps = self.val_steps,
                                                 initial_epoch = 0,
                                                 callbacks = [self.cp_cb, self.es_cb, self.tb_cb])
        return history.epoch

    def predict(self):
        path = self.base_dir + self.inf_dir
        for i, name in enumerate(self.inf_list):
            imgs = []
            img = cv2.imread(self.base_dir + self.img_dir + name)

            imgs.append(self.crop_resize_norm_bgr(img, self.input_shape))
            imgs = np.array(imgs)

            inference = self.multi_model.predict(imgs)
            out = cv2.resize(inference[0], (1024, 256))

            print('\r\033[1A\033[0KInference done on %d of %d Images' % (i, len(self.inf_list)))
            cv2.imwrite(path + name, out * 255)

    def finish(self):
        K.clear_session()

    def update_callback(self):
        print('saving weights in %s' % self.log_dir)
        # set callbacks
        self.cp_cb = ModelCheckpoint(
            filepath = self.log_dir + '/weights{epoch:02d}.hdf5',
            # filepath = path + '/weights{val_loss:02d}.hdf5',
            monitor = 'val_loss',
            verbose = 1,
            save_best_only = True,
            mode = 'auto',
            period = 1)
        self.es_cb = EarlyStopping(
            monitor = 'val_loss',
            patience = 4,
            verbose = 1,
            mode = 'auto')
        self.tb_cb = TensorBoard(
            log_dir = self.log_dir,
            write_images = True)
