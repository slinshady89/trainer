import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from trainer import Trainer

from random import shuffle



def main():
    eval_base_dir = '/home/nils/nils/results/'
    eval_img_dir = 'image_2/'
    inf_dir = 'pooling_test/'
    # /absolute/directory/to/weightsXX.hdf5
    weights_dir = ''
    load_weights = True

    test_no_indices = True  # True if model without indice forwarding should be loaded
    segnet = False  # True if SegNet model should be loaded
    if segnet:
        inf_dir_tested = 'SegNet/'
        gpu_num = '2, 3'
    else:
        inf_dir_tested = 'test/'
        gpu_num = '2'

    # list that contains all images for the inference
    eval = sorted(os.listdir(eval_base_dir + eval_img_dir))
    print('\n%d images for inference available.\n' % eval)

    with tf.Graph().as_default():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        trainer = Trainer(_inf_list = eval,
                          _gpu_num = gpu_num,
                          _no_inidices = test_no_indices,
                          _segnet = segnet,
                          _load_weights = load_weights,
                          _weights_dir = weights_dir)

        # directory images are loaded trainer.base_dir + trainer.inf_dir
        trainer.base_dir = eval_base_dir
        trainer.inf_dir = inf_dir + inf_dir_tested

        trainer.base_dir = eval_base_dir
        trainer.img_dir = eval_img_dir
        print('Loading imgs from %s' % (trainer.base_dir + trainer.img_dir))

        # trainer.predict()
        trainer.predict_own()


if __name__ == "__main__":
    try:
        main()
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\nCancelled by user. \n\nGoodbye!")
