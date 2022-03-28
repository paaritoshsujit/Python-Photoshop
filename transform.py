
#####################################################################################
##### Implemtation of the Python Photo Editing, and Creation of and edge detector####
#####################################################################################

from image import Image
import numpy as np



def brighten(image, factor):
    # when we brighten, we just want to make each channel higher by some amount (factor)
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)
    
    # first we find the size of the image, so that we can iterate through each pixel
    x_pixels, y_pixels, num_channels = image.array.shape    

    # we create a new, empty image so that we don't mutate the original one
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)

    # # the non-vectorised method is the most intuitive way to perform the edit
    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             new_image.array[x, y, c] = image.array[x ,y ,c] * factor

    # vetorised version, much easier to write. it leverages numpy
    new_image.array  = image.array * factor

    return new_image




def adjust_contrast(image, factor, mid):
    # adjust the contrast by increasing the difference from the user-defined midpoint by factor amount
    x_pixels, y_pixels, num_channels = image.array.shape
    new_image = Image(x_pixels = x_pixels, y_pixels = y_pixels, num_channels = num_channels)
    
    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             # we find the difference b/w the midpoint and the actual value, and amplify that difference to increase contrast
    #             new_image[x, y, c] = (image.array[x, y, c] - mid) * factor + mid

    # vectorised version, leveraging numpy 
    new_image.array = (image.array - mid) * factor + mid

    return new_image



def blur(image, kernel_size):
    # kernel size is the number of pixels to take into account when applying the blur
    # (ie kernel_size = 3 would be neighbors to the left/right, top/bottom, and diagonals)
    # kernel size should always be an *odd* number

    x_pixels, y_pixels, num_channels = image.array.shape
    new_image = Image(x_pixels = x_pixels, y_pixels = y_pixels, num_channels = num_channels)

    neighbour_range = kernel_size // 2   # how many neighbours to one side we need to look at

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                # here we employ a naive implementation og iterating through each neighbour and summing values
                total = 0 
                for x_i in range(max(0, x- neighbour_range), min(x_pixels-1, x+ neighbour_range)+1):    # max and min is used to check for edge pixels
                    for y_i in range(max(0, y-neighbour_range), min(y_pixels-1, y+ neighbour_range)+1):
                        total += image.array[x_i, y_i, c]
                
                new_image.array[x, y ,c] = total/(kernel_size ** 2)

    return new_image
    

def apply_kernel(image, kernel):
    # the kernel should be a numpy 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]
    x_pixels, y_pixels, num_channels = image.array.shape
    new_image = Image(x_pixels = x_pixels, y_pixels = y_pixels, num_channels = num_channels)

    kernel_size = kernel.shape[0]
    neighbour_range = kernel_size // 2   # how many neighbours to one side we need to look at

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0 
                for x_i in range(max(0, x-neighbour_range), min(x_pixels-1, x + neighbour_range)+1):
                    for y_i in range(max(0, y - neighbour_range), min(y_pixels -1 , y + neighbour_range)+ 1):
                        # we need to find which value of the kernel this corersponds to
                        x_k = x_i + neighbour_range - x
                        y_k = y_i + neighbour_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                
                new_image.array[x, y, c] = total
    
    return new_image



def combine_images(image1, image2):
    # let's combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)
    # size of image1 and image2 MUST be the same
    x_pixels, y_pixels, num_channels = image1.array.shape
    new_image = Image(x_pixels = x_pixels, y_pixels = y_pixels, num_channels = num_channels)

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_image.array[x, y, c] = (image1.array[x,y,c]**2 + image2.array[x,y,c]**2)**0.5

    return new_image

    



if __name__ == '__main__':
    lake = Image(filename='lake.png')
    city = Image(filename='city.png')

    # let's lighten the lake and darken the city
    brightened_img = brighten(lake, factor = 1.9)
    brightened_img.write_image('brightened.png')

    darkened_img = brighten(city, factor = 0.4)
    darkened_img.write_image('darkened.png')


    # lets now increase the contrast of the lake and decrease for the city
    incr_contrast = adjust_contrast(lake, 2, 0.5)
    incr_contrast.write_image('increased_contrast.png')

    dec_contrast = adjust_contrast(city, 0.7, 0.5)
    dec_contrast.write_image('decreased_contrast.png')


    # lets blur the city with kernel sizes 3 and 9
    blur1 = blur(city, 3)
    blur1.write_image('blur1.png')

    blur2 = blur(city, 9)
    blur2.write_image('blur2.png')


    # lets apply a sobel edge detection kernel on the x and y axis to the city
    sobel_x_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    sobel_y_kernel = np.array([
        [1, 0, -1], 
        [2, 0, -2], 
        [1, 0, -1]
    ])

    sobel_x = apply_kernel(city, sobel_x_kernel)
    sobel_x.write_image('edge_x.png')

    sobel_y = apply_kernel(city, sobel_y_kernel)
    sobel_y.write_image('edge_y.png')


    # lets combine the two sobel images and make an edge detector!
    sobel_xy = combine_images(sobel_x, sobel_y)
    sobel_xy.write_image('edge_xy.png')
