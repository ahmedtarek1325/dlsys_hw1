import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(label_filename, 'rb') as f:
        magic,size= struct.unpack(">II",f.read(8))
        y = np.array(list(f.read()),dtype=np.uint8)
        y = y.reshape((size,)) 
    with gzip.open(image_filesname,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        X = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        X = X.reshape((size, nrows*ncols))
        X=X.astype(np.float32)
        X/=255
    return X,y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    
    zy= ndl.summation(ndl.multiply(Z,y_one_hot),axes= 1 )
    zy=zy.reshape((-1,1))
    zy= ndl.broadcast_to(zy,Z.shape)
    return ndl.summation(ndl.log(ndl.summation(ndl.exp((Z-zy)),axes= 1)))/Z.shape[0]
    '''
    loss = ndl.log(ndl.summation(ndl.exp(Z),axes = 1)) - ndl.summation(Z*y_one_hot,axes = 1)
    return ndl.summation(loss)/Z.shape[0]
    '''
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    i,end = 0, X.shape[0]
    while i < end: 
        index = i+batch if (i+batch) <= end else end 

        l1_output=  ndl.Tensor(X[i:index]) @ W1
        Z1 = ndl.relu(l1_output)
        outputs = Z1 @ W2 # DIM = m*K
        
        y_one_hot= np.zeros(outputs.shape)
        for inde,val in enumerate(y[i:index]): y_one_hot[inde,val]= 1
        
        out = softmax_loss(outputs,ndl.Tensor(y_one_hot))
        out.backward()

        
        
        W1 = W1.numpy() - lr*W1.grad.numpy()
        W2 = W2.numpy() - lr*W2.grad.numpy()

        W1,W2 = ndl.Tensor(W1),ndl.Tensor(W2)
        i+=batch



    return (W1,W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
