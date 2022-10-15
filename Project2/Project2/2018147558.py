import numpy as np
import cv2
import sys
import os


#for number of dimensions required ratio(step1)
def select_dms(s, percent):
    idx = 0
    temp = 0
    while percent > temp :
        temp += s[idx] / np.sum(s)
        idx += 1
    return idx

#for reconstruction of the image(step2)
def img_reconst(face, Vt, path, train_list, mean, idx):
    #svd
    norm_face = face - mean
    temp = np.dot(norm_face, Vt.T)
    vector = np.dot(temp[idx,:], Vt)
    reconst = mean + vector
    
    allsum = (reconst - face[idx]) ** 2
    allsum = np.sum(allsum) / (h * w)
    
    #write image
    wpath = path + "/" + train_list
    cv2.imwrite(wpath, reconst.reshape(h,w))
    
    return allsum

#for test_
def vector_re(face, Vt, mean):
    #svd
    norm_face = face - mean
    temp = np.dot(norm_face, Vt.T)
    
    return temp

#for reading a set of images
def read_img(path, nimg):
    img_list = os.listdir(path)[0:nimg]
    image = [path+"/"+ entry for entry in img_list]
    images = np.array([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image],
                          dtype = np.float64)
    n, h, w = images.shape
    images = images.reshape(n, h*w)
    return images, img_list, h, w

#for writing results to txt files
def text_write(percent, slt_dms, error_arr, result, test_list, train_list):
    
    ##################### step1 #####################
    file_out = open('2018147558/output.txt','w')
    file_out.write("########## STEP 1 ##########\n")
    file_out.write("Input Percentage: %s" %percent)
    file_out.write("\nSelected Dimension: %d\n" %slt_dms)
    
    ##################### step2 #####################
    avg_error = sum(error_arr) / len(error_arr)
    file_out.write("\n########## STEP 2 ##########\n")
    file_out.write("Reconstruction error\n")
    file_out.write("Average: {:.4f}\n".format(avg_error))
    
    for i in range(39):
        line = format(i+1, '02') + ": " + "{:.4f}\n".format(error_arr[i])
        file_out.write(line)
    
    ##################### step3 #####################
    file_out.write("\n########## STEP 3 ##########\n")
    
    for i in range(0,5):
        line = test_list[i] + " ==> " + train_list[result[i]] + "\n"
        file_out.write(line)

#Variables that need to be initialization
percent = sys.argv[1]
train_path = 'faces_training'
test_path = 'faces_test'
s2_path = '2018147558'
error_arr = []
result = []

#main function
if __name__ == '__main__':

    #When there is an error in the factor given to the main function
    if len(sys.argv) != 2 :
        print("Invalid arguments")
        sys.exit()
    
    #Read train and test image set
    train_img, train_list, h, w = read_img(train_path, 39)
    test_img, test_list, h, w = read_img(test_path, 5)
    
    #Average face value for zero mean
    mean = np.mean(train_img, axis=0)
    
    #apply svd to train image
    U, s, Vt = np.linalg.svd(train_img-mean, full_matrices = False)
    s *= s
    
    #step1.
    slt_dms = select_dms(s, float(percent))
    img_vector = np.ndarray(shape=(39,slt_dms))
    
    #step2.
    for i in range(39):
        error = img_reconst(train_img, Vt[:slt_dms], s2_path, train_list[i], mean, i)
        img_vector[i] = vector_re(train_img[i], Vt[:slt_dms], mean)
        error_arr.append(error)
    
    #Step 3
    for i in range(0,5):
        test_vector = vector_re(test_img[i], Vt[:slt_dms], mean)
        dit = np.sqrt(np.sum((img_vector - test_vector)**2, axis = 1))
        result.append(np.argmin(dit))
    
    #output 
    text_write(percent, slt_dms, error_arr, result, test_list, train_list)

    

    
    
    
    
    