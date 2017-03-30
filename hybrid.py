# -*- coding: utf-8 -*-
import numpy as np
import cv2
# '''
# visualize a hybrid image by progressively downsampling the image and
# concatenating all of the images together.
# '''
def vis_hybrid_image(hybrid_image):

    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    
    output = hybrid_image
    cur_image = hybrid_image

    for i in range(2, scales + 1):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)
        cur_image = cv2.resize(cur_image, (0,0), fx=scale_factor, fy=scale_factor)
        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)
    
    return output

def my_imfilter(image, kernel, padding_type):
#    TODO 1: implement your own image filter

    k_r = kernel.shape[0]
    k_c = kernel.shape[1]

    r = image.shape[0]
    c = image.shape[1]

    p = (k_r-1)/2 #padding border
#   padding_type 0: zero padding
    if padding_type == 0:                        
        image = cv2.copyMakeBorder(image, p,p,p,p, cv2.BORDER_CONSTANT, value=0)
#   padding_type 1: replicate
    elif padding_type == 1:
        image = cv2.copyMakeBorder(image,p,p,p,p,cv2.BORDER_REPLICATE)
#   padding_type 2: symetric
    elif padding_type == 2:
        image = cv2.copyMakeBorder(image,p,p,p,p,cv2.BORDER_REFLECT)


    #create empty image to store output
    output = np.zeros((r, c, 3), np.uint8)

    #handle RGB
    for y in np.arange(p, r + p):
        for x in np.arange(p, c + p):
            roi_red   = image[y-p : y+p+1, x-p : x+p+1, 2]*256
            roi_green = image[y-p : y+p+1, x-p : x+p+1, 1]*256
            roi_blue  = image[y-p : y+p+1, x-p : x+p+1, 0]*256
            output[y-p,x-p,2] = (roi_red * kernel).sum()
            output[y-p,x-p,1] = (roi_green * kernel).sum()
            output[y-p,x-p,0] = (roi_blue * kernel).sum()

    return output



if __name__ == '__main__':
    image1 = cv2.imread('..\data\dog.bmp') /255.0
    image2 = cv2.imread('..\data\cat.bmp') /255.0

    cutoff_frequency = 7

    kernel = cv2.getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency)
    kernel = cv2.mulTransposed(kernel, False)

    """
    YOUR CODE BELOW. Use my_imfilter to create 'low_frequencies' and
    'high_frequencies' and then combine them to create 'hybrid_image'
    """


    """
    TODO 2:
    Remove the high frequencies from image1 by blurring it. The amount of
    blur that works best will vary with different image pairs
    """
    low_frequencies = my_imfilter(image1, kernel,0)

    """
    TODO 3:
    Remove the low frequencies from image2. The easiest way to do this is to
    subtract a blurred version of image2 from the original version of image2.
    This will give you an image centered at zero with negative values.
    """

    high_frequencies = image2*255.0 - my_imfilter(image2, kernel,0) + 0.5

    """
    TODO 4: 
    Combine the high frequencies and low frequencies
    """

    hybrid_image = low_frequencies + high_frequencies

    """
    Visualize and save outputs
    """
    vis = vis_hybrid_image(hybrid_image)

    cv2.imshow("low", low_frequencies)
    cv2.imshow("high", high_frequencies)
    cv2.imshow("vis", vis)
    cv2.waitKey(0)

    cv2.imwrite('low_frequencies.jpg', low_frequencies)
    cv2.imwrite('high_frequencies.jpg', high_frequencies)
    cv2.imwrite('hybrid_image.jpg', hybrid_image)
    cv2.imwrite('hybrid_image_scales.jpg', vis)