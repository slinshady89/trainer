import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from trainer import Trainer

from random import shuffle



def main():
    base_dir = '/media/localadmin/Test/11Nils/kitti/dataset/sequences/Data/'
    label_dir = 'labels/'
    train_img_dir = 'images/'
    eval_base_dir = '/media/localadmin/Test/11Nils/kitti/dataset/sequences/08/'
    eval_lbl_dir = 'labels/'
    eval_img_dir = 'image_2/'
    inf_dir = 'pooling_test/'
    test_no_indices = True
    if test_no_indices:
        inf_dir_tested = 'MaxPooling2D/'
        log = 'log/'
        gpu_num = '0, 1, 2, 3'
    else:
        inf_dir_tested = 'MaxPooling2DWithIndices/'
        log = 'log_indices/'
        gpu_num = '2, 3'
    segnet = True
    if segnet:
        inf_dir_tested = 'SegNet/'
        log = 'log_seg/'
        gpu_num = '2, 3'

    label_list = sorted(os.listdir(base_dir + label_dir))

    shuffle(label_list)

    train = label_list[:int(len(label_list)*0.8)]
    val = label_list[int(len(label_list)*0.8):]
    eval = os.listdir(eval_base_dir + eval_lbl_dir)

    print(len(train), len(val), len(eval))

    with tf.Graph().as_default():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        trainer = Trainer(_train_list = train,
                          _val_list = val,
                          _inf_list = eval,
                          _gpu_num = gpu_num,
                          _no_inidices = test_no_indices,
                          _segnet = segnet)
        trainer.base_dir = base_dir
        trainer.label_dir = label_dir
        trainer.img_dir = train_img_dir
        trainer.log_dir = eval_base_dir + inf_dir + log
        trainer.inf_dir = inf_dir + inf_dir_tested
        trainer.batch_size = 16
        trainer.epoch_steps = 750
        trainer.val_steps = 200
        trainer.n_epochs = 30
        trainer.dag_it = 0
        trainer.update_callback()
        # trains model for defined number of epochs with the actual dataset
        print('Loading labels from %s' % (trainer.base_dir + trainer.label_dir))
        print('Loading imgs from %s' % (trainer.base_dir + trainer.img_dir))
        trainer.train()
        print('\nTraining done!\nStarting Prediction\n')
        # safes inferences of images that are unseen by the net

        trainer.base_dir = eval_base_dir
        trainer.img_dir = eval_img_dir
        print('Loading labels from %s' % (trainer.base_dir + trainer.label_dir))
        print('Loading imgs from %s' % (trainer.base_dir + trainer.img_dir))

        trainer.predict()
        session.close()






if __name__ == "__main__":
    try:
        main()
        print("\nFinished without interrupt. \n\nGoodbye!")
    except KeyboardInterrupt:
        print("\nCancelled by user. \n\nGoodbye!")
