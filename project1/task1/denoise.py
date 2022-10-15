import cv2
import numpy as np

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    
    #Apply each filter to the noisy image
    temp1 = apply_bilateral_filter(noisy_img.copy(), 7, 12, 60)
    temp2 = apply_median_filter(noisy_img.copy(), 3)
    temp3 = apply_my_filter(noisy_img.copy(),3, 50)
    
    #rms for filtered images
    v1 = calculate_rms(clean_img,temp1)
    v2 = calculate_rms(clean_img,temp2)
    v3 = calculate_rms(clean_img,temp3)
    
    #Select the image with the smallest rms value as the result
    result_img = temp1
    if v2 < v1 : result_img = temp2
    if v3 < v2 : result_img = temp3
    
    print(calculate_rms(clean_img,result_img))
    
    # do noise removal
    
    cv2.imwrite(dst_path, result_img)
    
    return 0

def apply_median_filter(img, kernel_size):
    
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    #Receive information on the image.
    Height = img.shape[0]
    Width = img.shape[1]
    ch = img.shape[2]
    
    #Middle index of kernel
    index = kernel_size // 2
    
    #Copy body to retain the original image value
    img_copy = img.copy().tolist()
    for c in range(ch):
        for h in range(Height):
            for w in range(Width):
                    mid_mat = []
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            x = h-index+i
                            y = w-index+j
                            #Processing Boundary Values
                            if x < 0 or y < 0 : pass
                            elif x >= Height or y >= Width : pass
                            else :
                                mid_mat.append(img_copy[x][y][c])
                    #Process for obtaining intermediate values
                    mid_mat.sort();
                    idx = len(mid_mat) // 2
                    img[h][w][c] = mid_mat[idx]
                
    return img

#Application of Gaussian functions with one x value
def gauss(x, sigma):
    return 1/(sigma**2*2*np.pi) * np.exp(-(x*x)/(sigma**2*2))

def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    
    #Receive information on the image.
    Height = img.shape[0]
    Width = img.shape[1]
    ch = img.shape[2]
    
    #Middle index of kernel
    index = kernel_size // 2
    
    #Copy body to retain the original image value
    img_copy = img.copy().tolist()
    
    for h in range(Height):
        for w in range(Width):
            for c in range(ch):
                #The value of the pixel with the filter applied and the variable to normalize it
                ft_pixel = 0
                normal = 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        #Specify the scope to which the filter applies
                        hindex = h - index + i
                        windex = w - index + j
                        #Processing Boundary Values
                        if hindex >= Height : hindex -= Height
                        if windex >= Width : windex -= Width
                        #Gaussian operations for range values
                        rge = img_copy[hindex][windex][c] - img_copy[h][w][c]
                        rge = gauss(rge, sigma_r)
                        #Gaussian operations for distance values
                        dit = np.sqrt(((h-hindex)**2) + ((w-windex)**2))
                        dit = gauss(dit, sigma_s)
                        #Convolution two operations and obtaining pixe values
                        bilater = dit * rge
                        ft_pixel = ft_pixel + (img_copy[hindex][windex][c] * bilater)
                        normal = normal + bilater
                #Apply obtained pixel value
                ft_pixel = ft_pixel // normal
                ft_pixel = int(ft_pixel)
                img[h][w][c] = ft_pixel
                
    return img

#Functions that return Gaussian kernels that fit the kernel size
def gaussian(kernel_size,sigma_s):
    kernel = np.zeros((kernel_size,kernel_size))
    mid = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            temp = ((i-mid)**2) + ((i-mid)**2)
            kernel[i][j] = np.exp(-temp/(sigma_s**2*2))/(sigma_s**2*2)
    kernel = kernel/np.sum(kernel)
    return kernel

def apply_my_filter(img, kernel_size, sigma_s):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    #Receive information on the image.
    Height = img.shape[0]
    Width = img.shape[1]
    ch = img.shape[2]
    
    #Middle index of kernel
    mid = kernel_size // 2
    
    #Copy body to retain the original image value
    img_copy = img.copy().tolist()
    
    #Creating a kernel to apply Gaussian filters
    gauss = gaussian(kernel_size, sigma_s)
    
    for h in range(Height):
        for w in range(Width):
            for c in range(ch):
                #The value of the pixel with the filter applie
                pixel = 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        #Specify the scope to which the filter applies
                        x = h - mid + i
                        y = w - mid + j 
                        #Processing Boundary Values
                        if x >= Height : x -= Height
                        if y >= Width : y -= Width
                        #intens value for applying sharpness filter
                        intens = 0
                        #intens value for median value of filter
                        if x == h and y == w : intens = 2
                        #Applying and Normalizing Intens Values
                        temp = img_copy[x][y][c] * intens - (1/(kernel_size**2))
                        #Applying and Normalizing Intens Values
                        temp = temp * gauss[i][j]
                        #Adjusted Value
                        temp *= 4.3
                        pixel += temp
                img[h][w][c] = pixel
                
    return img


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))

