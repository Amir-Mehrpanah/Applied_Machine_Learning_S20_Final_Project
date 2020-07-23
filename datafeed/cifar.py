import tensorflow as tf

def get_data():
    (dstrain, _), (dstest, _) = tf.keras.datasets.cifar10.load_data()
    dstrain = dstrain.astype('float32').reshape(
        (dstrain.shape[0], 32, 32, 3)) / 255
    dstest = dstest.astype('float32').reshape(
        (dstest.shape[0], 32, 32, 3)) / 255
    
    print('train images:',len(dstrain))
    print('test images:',len(dstest))
    return dstrain, dstest