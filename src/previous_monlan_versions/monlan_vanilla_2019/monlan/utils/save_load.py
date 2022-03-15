import joblib

def save(path, obj, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))

    with open(path, "wb") as objFile:
        joblib.dump(obj, objFile)

    if verbose:
        print("Object saved to {}".format(path))
    pass

def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as objFile:
        obj = joblib.load(objFile)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj