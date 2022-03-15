
def useCpu(nThreads, nCores):
    import tensorflow as tf
    from keras import backend as K
    numThreads = nThreads
    num_CPU = nCores
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=numThreads,
                            inter_op_parallelism_threads=numThreads,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )
    session = tf.Session(config=config)
    K.set_session(session)

#full gpu memory clean
def reset_keras():
    import tensorflow
    from keras.backend.tensorflow_backend import set_session
    from keras.backend.tensorflow_backend import clear_session
    from keras.backend.tensorflow_backend import get_session

    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))