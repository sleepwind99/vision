import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    #Receive information on the image.
    height, width = img.shape
    centerh = height // 2
    centerw = width // 2
    
    #Copy body to retain the original image value
    img_copy = img.copy()
    
    for i in range(height):
        for j in range(width):
            #Normalized value index to the center of the image.
            x = i + centerh
            y = j + centerw
            if x >= height : x = x - height
            if y >= width : y= y - width
            #Apply to Image
            img[i][j] = img_copy[x][y]
            
    return img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    
    #Receive information on the image.
    height, width = img.shape
    centerh = height // 2
    centerw = width // 2
    
    #Copy body to retain the original image value
    img_copy = img.copy()
    
    
    for i in range(height):
        for j in range(width):
            #Normalized value index to the center of the image.
            x = i - centerh
            y = j - centerw
            if x >= height : x = x - height
            if y >= width : y= y - width
            #Apply to Image
            img[x][y] = img_copy[i][j]
            
    return img

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    #Fourier transform for images
    fqimg = np.fft.fft2(img)
    #Switching to Image Coordinates
    fqimg = fftshift(fqimg)
    #Processing to be a valid value
    mgtspt = 20 * np.log(np.abs(fqimg))
    
    return mgtspt

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    #Receive information on the image.
    height, width = img.shape
    centerh = height //2
    centerw = width // 2
    
    #Fourier transform for images
    img_spt = np.fft.fft2(img)
    
    for i in range(height):
        for j in range(width):
            #Normalized value index to the center of the image.
            x = i - centerh
            y = j - centerw
            if x >= height : x = x - height
            if y >= width : y = y - width
            #the equation of a circle
            if (x**2) + (y**2) >= (r**2) : img_spt[x][y] = 0
    
    #Reverse Fourier transform
    img_result = np.fft.ifft2(img_spt)
    img_result = np.abs(img_result)
    
    return img_result

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    #Receive information on the image.
    height, width = img.shape
    centerh = height //2
    centerw = width // 2
    
    #Fourier transform for images
    img_spt = np.fft.fft2(img)
    
    for i in range(height):
        for j in range(width):
            #Normalized value index to the center of the image.
            x = i - centerh
            y = j - centerw
            if x >= height : x = x - height
            if y >= width : y = y - width
            #the equation of a circle
            if (x**2) + (y**2) <= (r**2) : img_spt[x][y] = 0
    
    #Reverse Fourier transform
    img_result = np.fft.ifft2(img_spt)
    img_result = img_result.real
    
    return img_result

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    #Receive information on the image.
    height, width = img.shape
    centerh = height //2
    centerw = width // 2
    
    #Fourier transform for images
    img_spt = np.fft.fft2(img)
    
    for i in range(height):
        for j in range(width):
            #Normalized value index to the center of the image.
            x = i - centerh
            y = j - centerw
            if x >= height : x = x - height
            if y >= width : y= y - width
            
            #Fill the empty space through Fourier transform.
            if (51 < x and x < 57) and (51 < y and y < 57): 
                img_spt[x][y] = 0
                img_spt[x][-y] = 0
                img_spt[-x][y] = 0
                img_spt[-x][-y] = 0
            if (79 < x and x < 85) and (79 < y and y < 85): 
                img_spt[x][y] = 0
                img_spt[x][-y] = 0
                img_spt[-x][y] = 0
                img_spt[-x][-y] = 0
    
    #Reverse Fourier transform
    img_result = np.fft.ifft2(img_spt)
    img_result = np.abs(img_result)

    return img_result

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    #Receive information on the image.
    height, width = img.shape
    centerh = height //2
    centerw = width // 2
    
    #Fourier transform for images
    img_spt = np.fft.fft2(img)
    
    for i in range(height):
        for j in range(width):
            #Normalized value index to the center of the image.
            x = i - centerh
            y = j - centerw
            if x >= height : x = x - height
            if y >= width : y= y - width
            #Fill the empty space through Fourier transform.
            if (x**2) + (y**2) < 800 and (x**2) + (y**2) > 720: img_spt[x][y] = 0
    
    #Reverse Fourier transform
    img_result = np.fft.ifft2(img_spt)
    img_result = np.abs(img_result)
                
    return img_result

#################

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()