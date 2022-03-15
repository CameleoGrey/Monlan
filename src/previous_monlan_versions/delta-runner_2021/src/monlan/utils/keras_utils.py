
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
