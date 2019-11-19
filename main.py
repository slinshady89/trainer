import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from trainer import Inferencer


def main():

    # /absolute/directory/to/images/for/inference/
    inf_dir = '/media/Test/00_Nils/kitti/data/sequences/08/image_2/'
    # /absolute/directory/to/weightsXX.hdf5
    weights_dir = ''
    load_weights = True

    test_no_indices = True  # True if model without indice forwarding should be loaded
    segnet = False  # True if SegNet model should be loaded

    gpu_num = '2'

    # list that contains all images for the inference
    inf_imgs = sorted(os.listdir(inf_dir))
    print('\n%d images for inference available.\n' % inf_imgs)

    with tf.Graph().as_default():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        trainer = Inferencer(_inf_list = inf_imgs,
                             _gpu_num = gpu_num,
                             _no_inidices = test_no_indices,
                             _segnet = segnet,
                             _load_weights = load_weights,
                             _weights_dir = weights_dir)
        # directory images are loaded trainer.base_dir + trainer.inf_dir
        trainer.inf_dir = inf_dir
        print('Loading imgs from %s' % trainer.inf_dir)

        # trainer.predict()
        trainer.predict_own()


if __name__ == "__main__":
    try:
        main()
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\nCancelled by user. \n\nGoodbye!")
