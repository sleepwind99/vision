import cv2
import numpy as np
import random

## Do not erase or modify any lines already written
## Each noise function should return image with noise

def add_gaussian_noise(image):
    # Use mean of 0, and standard deviation of image itself to generate gaussian noise
    # Use the shape function to obtain the height and width values of the image.
    
    Height = image.shape[0]
    Width = image.shape[1]
    
    # Variable that amplifies the noise value
    intensity = 28.3
    # for loop to apply noise per pixel
    
    for h in range(Height):
        for w in range(Width):
            # function of Numpy that randomly returns normal distribution information values
            random_normal_value = np.random.normal()
            noise = intensity * random_normal_value
            # Add the generated noise value to the original image and store it.
            image[h][w] = image[h][w] + noise
            
    return image

def add_uniform_noise(image):
    # Generate noise of uniform distribution in range [0, standard deviation of image)
    # Use the shape function to obtain the height and width values of the image.
    
    Height = image.shape[0]
    Width = image.shape[1]
    
    # Variable that amplifies the noise value
    intensity = 29
    # for loop to apply noise per pixel
    
    for h in range(Height):
        for w in range(Width):
            # function of Numpy to obtain random equal distribution values
            random_uniform_value = np.random.uniform()
            noise = intensity * random_uniform_value
            # Add the generated noise value to the original image and store it.
            image[h][w] = image[h][w] + noise
    return image

def apply_impulse_noise(image):
    # Implement pepper noise so that 20% of the image is noisy
    # Use the shape function to obtain the height and width values of the image.
    
    Height = image.shape[0]
    Width = image.shape[1]
    
    # adjustment of noise strength and weakness(0~1)
    Criteria = 0.0776
    # for loop to apply noise per pixel
    
    for h in range(Height):
        for w in range(Width):
            # function for obtaining random values from 0 to 1
            random_value = random.random()
            if random_value < Criteria :
                # The smallest value in color
                image[h][w] = 0
            elif random_value > (1-Criteria) :
                # The color value is 8 bits.
                image[h][w] = 255
    return image

    

def rms(img1, img2):
    # This function calculates RMS error between two grayscale images. 
    # Two images should have same sizes.
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))

    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    np.random.seed(0)
    original = cv2.imread('bird.jpg', cv2.IMREAD_GRAYSCALE)
    
    gaussian = add_gaussian_noise(original.copy())
    print("RMS for Gaussian noise:", rms(original, gaussian))
    cv2.imwrite('gaussian.jpg', gaussian)
    
    uniform = add_uniform_noise(original.copy())
    print("RMS for Uniform noise:", rms(original, uniform))
    cv2.imwrite('uniform.jpg', uniform)
    
    impulse = apply_impulse_noise(original.copy())
    print("RMS for Impulse noise:", rms(original, impulse))
    cv2.imwrite('impulse.jpg', impulse)
