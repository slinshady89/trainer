import cv2
import os
import numpy as np
import time
from keras.preprocessing.image import img_to_array
from keras.utils import multi_gpu_model, print_summary
from keras.models import load_model
from layers import MaxUnpooling2D, MaxPoolingWithArgmax2D


class Inferencer(object):
    def __init__(self, _inf_list, _gpu_num = '0, 1',
                 _no_inidices = True, _segnet = False, _load_weights = False, _weights_dir = ''):
        self.inf_list = _inf_list
        self.inf_dir = ''
        self.gpu_num = _gpu_num  # '1'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # self.gpu_num
        if _no_inidices is True:
            self.multi_model = load_model(_weights_dir)
        else:
            self.multi_model = load_model(_weights_dir,
                                          custom_objects =
                                          {'MaxUnpooling2D': MaxUnpooling2D,
                                           'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D}
                                          )

        print(self.multi_model.summary())
        # self.multi_model.compile()
        # plot_model(model = self.multi_model, to_file = self.base_dir + 'model.png')
        # print(print_summary(self.multi_model))
        self.std = [0.32636853, 0.31895106, 0.30716496]
        self.mean = [0.39061851, 0.38151629, 0.3547171]

    def crop_resize_norm_bgr(self, img, dims):
        h, w, c = img.shape
        cropped_img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_img = cv2.resize(cropped_img, (dims[1], dims[0]))
        normed = resized_img / 255.0
        mean_free = (normed[:, :] - self.mean) / self.std
        return img_to_array(mean_free)

    def crop(self, img, dims):
        h, w, c = img.shape
        cropped_img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        return cv2.resize(cropped_img, (dims[1], dims[0])) / 255

    def crop_resive_mask(self, mask, dims):
        h, w, c = mask.shape
        cropped_mask = mask[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        resized_mask = cv2.resize(cropped_mask, (dims[1], dims[0]))
        return img_to_array(resized_mask) / 255

    def predict_Kitti(self):
        path = self.inf_dir
        for i, name in enumerate(self.inf_list):
            start = time.time()
            imgs = []
            img = cv2.imread(path + name)
            #imgs.append(self.crop_resize_norm_bgr(img, (1024, 256)))
            imgs.append(self.crop_resize_norm_bgr(img, (256, 1024)))
            # imgs.append(self.crop(img, self.input_shape))
            imgs = np.array(imgs)

            inference = self.multi_model.predict(imgs)
            out = cv2.resize(inference[0], (1024, 256))

            input = self.crop(img, (256, 1024))
            cv2.imshow('test', cv2.addWeighted(input, 0.5, np.asarray(out, np.float64), 0.5, 0.0))
            cv2.waitKey(10)
            elapsed = time.time() - start
            print('\r\033[1A\033[0KInference done on %d of %d Images at %.2f Hz' % (i, len(self.inf_list), 1 / elapsed))
            # cv2.imwrite(path + name, out * 255)

    def predict_own(self):
        path = self.inf_dir
        for i, name in enumerate(self.inf_list):
            start = time.time()
            imgs = []
            img = cv2.imread(path + name)
            if img is None:
                pass
            # imgs.append(self.crop_resize_norm_bgr(img, self.input_shape))
            imgs.append(self.crop_resize_norm_bgr(img, (256, 1024)))
            # imgs.append(img_to_array(cv2.resize(img / 255., (1024, 256))))
            imgs = np.array(imgs)

            inference = self.multi_model.predict(imgs)
            out = cv2.resize(inference[0], (1024, 256))
            input = cv2.resize(imgs[0], (1024, 256))
            cv2.imshow('overlaid', cv2.addWeighted(np.asarray(input, np.float64), 0.5,
                                                   np.asarray(out, np.float64), 0.5, 0.0))
            cv2.waitKey(10)
            elapsed = time.time() - start
            print('\r\033[1A\033[0KInference done on %d of %d Images at %.2f Hz' % (i, len(self.inf_list), 1 / elapsed))
