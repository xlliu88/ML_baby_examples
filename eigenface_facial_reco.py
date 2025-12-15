## a baby sample for facial recognizaitonusing eigenfaces

import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class EigenFacesResult:
    """    
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(map['subject_1_eigen_faces'][0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(map['subject_2_eigen_faces'][0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_1_mean_face: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        subject_2_mean_face: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.subject_1_mean_face = subject_1_mean_face
        self.subject_2_mean_face = subject_2_mean_face
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22


class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
            The path to the root directory containing subject images.
        """
        self.images_root_directory = images_root_directory
        
    def load_images(self, files, resize_factor):
        '''
        files: file names, support wildcard
        resize_factor: factor to resize. resize to 1/resize_factor

        return:
            an array of image pixels. (n, m)
                n: number of images
                m: vectorized image

        '''
        
        full_path = f'{self.images_root_directory}/{files}'
        fs = glob.glob(full_path)
        try:
            imgs = [Image.open(f) for f in fs]
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
        
        imgs_s = [img.resize((img.size[0]//resize_factor, img.size[1]//resize_factor)) for img in imgs]

        return np.array(imgs_s)
       
    def get_eigenface(self, imgs, ndim):
        '''
        images: matrix/tensor of images; images represented as 2/3d arrays
        ndim: dimension to reduced to

        return:
            dimensional reduced images; as same shape of the input image array
            mean face of pca
        '''

        shape = imgs.shape  ## (10, 60, 80)
        img2d = imgs.reshape(shape[0], np.prod(shape[1:]))  #(10, 4800)
        pca = PCA(n_components=ndim).fit(img2d)               # (ndim, 4800)
        eigenface = pca.components_.reshape(ndim, *shape[1:]) # (ndim, 60, 80)
        mean_face = pca.mean_.reshape(shape[1:])              # (60, 80)
        return eigenface, mean_face

    def get_ss(self, test_images, eigenfaces):
        '''
        test m test_images against k eigenfaces

        test_images: 2d array (m,n), each row as vectorized and centered test image;
        eigenfaces: 3d array (k, a, b);  
            k: # of objects
            a: # of eigenfaces (components)
            b: # of features in each eigenface
        return:
            2d array (k,n). 
            the (i,j)th item will be the projection residual of jth test image against ith eigenface
        '''

        m,n = test_images.shape
        k, a, b = eigenfaces.shape
        ss = np.zeros((k, m))
        # test_images_mean = test_images.T.mean(axis = 0)
        # test_images_c = (test_images.T - test_images_mean)

        for i in range(k): ## loop through eigenfaces
            ef = eigenfaces[i, :, :]
            for j in range(m):
                test_img = test_images[j,:]
                S = test_img.T - ef.T @ ef @ test_img.T
                norms = np.linalg.norm(S, axis = 0)
                ss[j, i] = norms**2
            
        return ss
        
    def run(self, ndim = 6) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """
        
        sub1 = self.load_images('subject01.*.gif', 4)
        sub2 = self.load_images('subject02.*.gif', 4)
        sub1t = self.load_images('subject01-test.gif', 4)
        sub2t = self.load_images('subject02-test.gif', 4)
        sub1t = sub1t.reshape(sub1t.shape[0], np.prod(sub1t.shape[1:]))
        sub2t = sub2t.reshape(sub2t.shape[0], np.prod(sub2t.shape[1:]))
        
        eigenfaces_1, mean_face1 = self.get_eigenface(sub1, ndim = ndim) # eigenfaces: (ndim, w, h); stack of ndim eigenfaces
        eigenfaces_2, mean_face2 = self.get_eigenface(sub2, ndim = ndim) # mean_faces:  # shaped the same as the input image
        shape1 = eigenfaces_1.shape
        shape2 = eigenfaces_2.shape
        
        ef1 = eigenfaces_1.reshape(shape1[0], np.prod(shape1[1:]))
        ef2 = eigenfaces_2.reshape(shape2[0], np.prod(shape2[1:]))

        test_images = np.vstack((sub1t, sub2t))
        eigenfaces = np.stack((ef1, ef2), axis = 0)
        meanfaces = np.vstack((mean_face1.reshape(-1), mean_face2.reshape(-1)))
        test_images_c = test_images - meanfaces
        ss_result = self.get_ss(test_images_c, eigenfaces)
        
        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            subject_1_mean_face=mean_face1,
            subject_2_mean_face=mean_face2,
            s11=ss_result[0,0],
            s12=ss_result[0,1],
            s21=ss_result[1,0],
            s22=ss_result[1,1]
        )


if __name__ == "__main__":
    EigFace = EigenFaces()
    result = EigFace.run()

    mean_face1 = result.subject_1_mean_face
    mean_face1 = mean_face1.reshape(1, *mean_face1.shape)
    mean_face2 = result.subject_2_mean_face
    mean_face2 = mean_face2.reshape(1, *mean_face2.shape)

    ## plot eigenfaces
    sub1_ef = np.concatenate((result.subject_1_eigen_faces, mean_face1), axis=0) # result.subject_1_mean_face))
    sub2_ef = np.concatenate((result.subject_2_eigen_faces, mean_face2), axis = 0) # result.subject_2_mean_face))

    fig, axes = plt.subplots(2,7,figsize = (14,4))
    for i, (ax1, ax2) in enumerate(axes.T):
        ax1.imshow(sub1_ef[i], cmap = 'gray')
        ax2.imshow(sub2_ef[i], cmap = 'gray')
        ax1.axis('off')
        ax2.axis('off')
    axes[0,0].text(x = 0.1, y = 1, s = 'subject 1 eigenface', ha = 'left', color = 'b')
    axes[0,6].text(x = 0.1, y = 1, s = 'subject 1 mean face', ha = 'left', color = 'b')
    axes[1,0].text(x = 0.1, y = 1, s = 'subject 2 eigenface', ha = 'left', color = 'b')
    axes[1,6].text(x = 0.1, y = 1, s = 'subject 2 mean face', ha = 'left', color = 'b')
    plt.suptitle('First 6 eigenfaces of subject 1 and subject 2',
                 x = 0.02,
                 y = 0.95,
                 ha = 'left')

    plt.tight_layout()
    plt.show()

    ## Display projection residual
    # proj_residual = np.array((result.s11, result.s12, result.s21, result.s22)).reshape(2,2)
    print('=== Projection residuals: ')
    print(f'subject 1 test image vs. subject 1 eigen face: {result.s11.round()}')
    print(f'subject 1 test image vs. subject 2 eigen face: {result.s12.round()}')
    print(f'subject 2 test image vs. subject 1 eigen face: {result.s21.round()}')
    print(f'subject 2 test image vs. subject 2 eigen face: {result.s22.round()}')


