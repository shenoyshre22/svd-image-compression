import numpy as np
def compress_image_svd(image,k):
    #image here is taken as a 2d numpy array
    #grayscale image is reported back as a matrix of size m*n
    #k is the number of  singular values that gives like a value to the compression ratio, higher means better quality but less compression and so on
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    #we're keeping only top k singular values so we get moderate quality and good compression at the same time
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    #now we're reconstructing the compressed image
    compressed= U_k @ S_k @ Vt_k    
    return compressed
