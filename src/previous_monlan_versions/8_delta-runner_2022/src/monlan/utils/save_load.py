import pickle

def save(obj, path, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))

    with open(path, "wb") as obj_file:
        pickle.dump( obj, obj_file, protocol=pickle.HIGHEST_PROTOCOL )

    if verbose:
        print("Object saved to {}".format(path))
    pass

def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as obj_file:
        obj = pickle.load(obj_file)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj