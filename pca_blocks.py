import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy.linalg as linalg
#%matplotlib inline
import os, glob
from PIL import Image
import cv2
import copy
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours as nbs

IMG_SIZE = 256
CWD = os.getcwd()
'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
    This function reconstructs an image X_recon_img given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''

    #TODO: Enter code below for reconstructing the image X_recon_img    
    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    M_coef = np.dot(D_im.T, X_mean.T)
    tmp1 = c_im - np.repeat(M_coef.reshape(-1, 1), n_blocks**2, 1)
    X_blocks = np.dot(D_im, tmp1) + np.repeat(X_mean.reshape(-1,1), n_blocks**2, 1)
    X_blocks = X_blocks.T
    slide_window = int(X_mean.size ** 0.5)
    image = T.tensor4('image')
    neibs = nbs.images2neibs(image, neib_shape = (slide_window, slide_window))
    transToImage = nbs.neibs2images(neibs, neib_shape = (slide_window, slide_window), original_shape = (1,1,IMG_SIZE, IMG_SIZE))
    trans_func = theano.function([neibs], transToImage)
    X_recon_img = trans_func(X_blocks)
    return X_recon_img[0,0]

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num), cmap = 'gray')
    os.chdir(CWD)        
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)
    
def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    image = T.tensor4('image')
    neibs = nbs.images2neibs(image, neib_shape = (sz, sz))
    transToImage = nbs.neibs2images(neibs, neib_shape = (sz, sz), original_shape = (1,1, sz, sz))
    trans_func = theano.function([neibs], transToImage)
    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(trans_func(D[:,[i*4+j]].T)[0,0], cmap = 'gray')
    os.chdir(CWD)
    f.savefig(imname)
    plt.close(f)
def getImgsFromFile():
     '''Read all images to the matrix X, where X is a Nx256x256 array, N is the number of images'''
     os.chdir(CWD+'/Fei_256')
     length = len([name for name in os.listdir('.') if os.path.isfile(name)])
     X = np.ndarray(shape = (length-1, IMG_SIZE, IMG_SIZE))
     i = 0
     for dirPath, dirNames, fileNames in os.walk(CWD + "/Fei_256"):
        for f in fileNames:
            if f.endswith('.jpg'):
                tmp = mpimg.imread(f, 0)
                X[i,:,:] = tmp
                i = i + 1
     return X
        
def getImgPatch(X, window):
    '''Return a tiled matrix X, where X is a number of images * blcoks_per_image by window ** 2 size matrix'''
    n = X.shape[0]
    image = T.tensor4('Image')
    '''create function that tile each image into a n_block_per_image ** 2 by window ** 2 size matrix'''
    neibs = nbs.images2neibs(image, neib_shape = (window, window))
    window_function = theano.function([image], neibs)
    X_blocks = None
    X_tmp = copy.copy(X)
    X_tmp.shape = (1, X_tmp.shape[0], X_tmp.shape[1], X_tmp.shape[2])
    X_blocks = window_function(X_tmp)
    return X_blocks
    
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    I = getImgsFromFile()
    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        ''' 
        #TODO: Write a code snippet that performs as indicated in the above comment
        X = getImgPatch(I , sz)
        '''unbiased estimation of covariance matrix = [sigma(x - u_x)(y - u_y)] / (n-1)
           n = # of images * blocks_per_image
        '''
        #S = np.cov(X)
        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)
        S = np.dot(X.T, X) / X.shape[0]
        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        #TODO: Write a code snippet that performs as indicated in the above comment
        v, D_ = linalg.eigh(S)
        D = D_[:,::-1]
        c = np.dot(D.T, X.T)
        os.chdir(CWD)
        for i in range(0, 200, 10):
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean, n_blocks=int(IMG_SIZE/sz), im_num=i)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()