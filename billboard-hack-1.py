# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from scipy.linalg import null_space


def histogram_eq(I):
    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # create the enhanced image
    J = np.zeros((I.shape[0],I.shape[1])).astype(int)

    # produce the histogram
    histogram = np.zeros(256).astype(int)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            histogram[I[i,j]] += 1

    # set up the cdf
    cdf = np.zeros(256)
    cdf[0] = histogram[0]/(I.shape[0]*I.shape[1])
    for i in range(1,256):
        cdf[i] = cdf[i-1]+histogram[i]/(I.shape[0]*I.shape[1])

    # scale the pixel values based on their cdf 
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            J[i,j] = round(cdf[I[i,j]]*255)

    return J


def dlt_homography(I1pts, I2pts):
    A = []

    # Check if two images have the same dimension
    if I1pts.shape != I2pts.shape:
        raise RuntimeError('The dimensions of the two images do not match.')

    # use list instead of ndarray
    for i in range(I1pts.shape[1]):
        A_temp_1 = [-I1pts[0][i], -I1pts[1][i], -1, 0, 0, 0, I1pts[0][i]*I2pts[0][i], I1pts[1][i]*I2pts[0][i], I2pts[0][i]]
        A_temp_2 = [0, 0, 0, -I1pts[0][i], -I1pts[1][i], -1, I1pts[0][i]*I2pts[1][i], I1pts[1][i]*I2pts[1][i], I2pts[1][i]]
        A.append(A_temp_1)
        A.append(A_temp_2)

    # convert list to ndarray
    A = np.asarray(A)

    # Compute the svd of A = U S V^T
    U, S, V = np.linalg.svd(np.asarray(A))

    # Find the column of V that correspond to the smallest sigular value
    H = V[-1].reshape((3,-1))
    
    # Normalize
    H /= H[-1,-1]
    return H, A


def bilinear_interp(I, pt):
    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # first check if the image is gray scaled
    if len(I.shape) != 2:
        raise RuntimeError('Image is not gray scale.')

    # get the points
    x = pt[0,0]
    y = pt[1,0]

    # locate the four points around (x,y)
    x1 = np.ceil(x).astype(int) - 1
    x2 = np.floor(x).astype(int) + 1
    y1 = np.ceil(y).astype(int) - 1
    y2 = np.floor(y).astype(int) + 1
    
    # make sure the pixel value will fall in the range of the image
    x1 = np.clip(x1, 0, I.shape[1] - 1)
    x2 = np.clip(x2, 0, I.shape[1] - 1)
    y1 = np.clip(y1, 0, I.shape[0] - 1)
    y2 = np.clip(y2, 0, I.shape[0] - 1)

    # two horizontal bilinear interpolation
    temp_1 = ((x2-x)/(x2-x1))*I[y1,x1] + ((x-x1)/(x2-x1))*I[y1,x2]
    temp_2 = ((x2-x)/(x2-x1))*I[y2,x1] + ((x-x1)/(x2-x1))*I[y2,x2]

    # bilinear interpolation in the vertical direction
    b = round(((y2-y)/(y2-y1))*temp_1 + ((y-y1)/(y2-y1))*temp_2)
        
    return b

def billboard_hack():
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    # [[x1,x2,x3,x4],[y1,y2,y3,y4]]
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('./yonge_dundas_square.jpg')
    Ist = imread('./uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    # Let's do the histogram equalization first.
    I_equ = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts,Ist_pts)

    path = Path(Iyd_pts.T)

    for i in range(min(bbox[0]), max(bbox[0])+1):
        for j in range(min(bbox[1]), max(bbox[1])+1):
            if path.contains_points([[i,j]]):
                # because H is 3-by-3 and pt is 2-by-1, create a homogeneous coord for x' = Hx
                x = np.array([[i],[j],[1]])
                # x' = Hx
                x_prime = H @ x
                # normalize
                x_prime /= x_prime[-1]
                # fill the image
                Ihack[j][i] = bilinear_interp(I_equ, x_prime[:-1,:])

    # calculate the reprojection error
    calculate_error(H)

    imwrite('./billboard_hacked_python.png', Ihack)

    plt.imshow(Ihack)
    plt.show()


def calculate_error(H):

    # 4-point correspondence from earlier.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    err = 0

    # compute the reprojection error for each set of correspondences.
    for i in range(4):
        x = np.array([[Iyd_pts[0,i]],[Iyd_pts[1,i]],[1]])
        x_prime = H @ x
        x_prime /= x_prime[-1]
        err += np.square(x_prime - np.array([[Ist_pts[0,i]],[Ist_pts[1,i]],[1]])).sum()

    print(err)
    

billboard_hack()
