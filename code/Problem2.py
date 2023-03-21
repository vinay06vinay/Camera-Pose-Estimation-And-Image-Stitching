import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
# from tqdm.notebook import tqdm
plt.rcParams['figure.figsize'] = [15, 15]
image1 = cv2.imread("image_1.jpg")
image2 = cv2.imread("image_2.jpg")
image3 = cv2.imread("image_3.jpg")
image4 = cv2.imread("image_4.jpg")
#Resize images due to computational limitations
image1_s = cv2.resize(image1,(0,0),fx = 0.2,fy=0.2, interpolation = cv2.INTER_AREA)
image2_s = cv2.resize(image2,(0,0),fx = 0.2,fy=0.2, interpolation = cv2.INTER_AREA)
image3_s = cv2.resize(image3,(0,0),fx = 0.2,fy=0.2, interpolation = cv2.INTER_AREA)
image4_s = cv2.resize(image4,(0,0),fx = 0.2,fy=0.2, interpolation = cv2.INTER_AREA)
image1_gray = cv2.cvtColor(image1_s, cv2.COLOR_RGB2GRAY)
image1_rgb = cv2.cvtColor(image1_s, cv2.COLOR_BGR2RGB)
image2_gray = cv2.cvtColor(image2_s, cv2.COLOR_RGB2GRAY)
image2_rgb = cv2.cvtColor(image2_s, cv2.COLOR_BGR2RGB)
image3_gray = cv2.cvtColor(image3_s, cv2.COLOR_RGB2GRAY)
image3_rgb = cv2.cvtColor(image3_s, cv2.COLOR_BGR2RGB)
image4_gray = cv2.cvtColor(image4_s, cv2.COLOR_RGB2GRAY)
image4_rgb = cv2.cvtColor(image4_s, cv2.COLOR_BGR2RGB)


def compute_homography(src_x,src_y,dest_x,dest_y):
    A=[]
    for i in range(len(src_x)):
        row1 = [src_x[i],src_y[i],1,0,0,0,-dest_x[i]*src_x[i],-dest_x[i]*src_y[i],-dest_x[i]]
        row2 = [0,0,0,src_x[i],src_y[i],1,-dest_y[i]*src_x[i],-dest_y[i]*src_y[i],-dest_y[i]]
        A.append(row1)
        A.append(row2)
    A= np.array(A)
    A_final = np.matmul(A.T , A)
    eig_values,eig_vectors = np.linalg.eig(A_final)
    eig_min_index = np.argmin(eig_values)
    eig_vector_min = np.array(eig_vectors[:,eig_min_index])
    H_eig  = eig_vector_min.reshape(3,3)
    H = (1/H_eig[2,2])*H_eig
    return H
    # U, s, V = np.linalg.svd(A)
    # H = V[-1].reshape(3, 3)
    # H = H * (1/H[2,2]) #To make sure the sum of all H values squared is equal to 1
    return H
def compute_error(H,src_x,src_y,dest_x,dest_y):
    #actual destination
    actual_dest = np.transpose(np.matrix([dest_x,dest_y,1]))
    actual_src = np.transpose(np.matrix([src_x,src_y,1]))
    estimated_dest = np.dot(H,actual_src)
    if(estimated_dest.item(2) == 0):
        return True
    error = actual_dest - ((1/estimated_dest.item(2))*estimated_dest )
    return np.linalg.norm(error)
    


def sift_matcher(img1,img2,image1_color,image2_color):
    img1 = cv2.GaussianBlur(img1, (11, 11),cv2.BORDER_DEFAULT)
    img2 = cv2.GaussianBlur(img2, (11, 11),cv2.BORDER_DEFAULT)
    # Create SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for match1,match2 in matches:
        if match1.distance < 0.5*match2.distance:
            good.append([match1])
    #Extracting all the points values derived from keypoint descriptors
    key_points_1 = np.float32([kp.pt for kp in kp1])
    key_points_2 = np.float32([kp.pt for kp in kp2])
    if len(good) > 4:
        # construct the two sets of points i.e get two points arrays from both the keypoint values which belong to the match array descriptors. So, getting only keypoints
        #in length to matches array
        pointsA = np.array(np.float32([key_points_1[m[0].queryIdx] for m in good]))
        pointsB = np.array(np.float32([key_points_2[m[0].trainIdx] for m in good]))
    pointsA_x = pointsA[:,0]
    pointsA_y = pointsA[:,1]
    pointsB_x = pointsB[:,0]
    pointsB_y = pointsB[:,1]
    # cv2.drawMatchesKnn expects list of lists as matches.
    sift_matches = cv2.drawMatchesKnn(image1_color,kp1,image2_color,kp2,good,None,flags=2)

    return (sift_matches,pointsA_x,pointsA_y,pointsB_x,pointsB_y)
def stitch_img(left, right, H):
    # Convert the image to float and normalize to make the image look more cleaner by removing high and low frequency noises
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    # left image
    left_height, left_width, left_channels = left.shape
    '''
    1. First we mark the corners in from left image. These corners are then transformed column wise to get corners in right image reference.
    '''
    corners = np.array([[0, 0, 1], [left_width, 0, 1], [left_width, left_height, 1], [0, left_height, 1]])
    corners_left = []
    for i in corners:
        corners_left.append(np.dot(H,i))
    corners_left = np.array(corners_left).T 
    #Getting the value of x and y in actual coordinates from homogenous coordinates of the corners of the transformed reference frame
    x_l_corners = corners_left[0] / corners_left[2]
    y_l_corners = corners_left[1] / corners_left[2]
    #Getting minimum of
    x = np.min(x_l_corners)
    y = np.min(y_l_corners)
    #Inverse transformation is now performed to get the left image size onto right one
    T_matrix = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    H = np.dot(T_matrix, H)
    #Modifying width so that two images could be stitched together
    height_new = int(round(abs(y) + left_height))
    width_new = int(round(abs(x) + left_width))
    size = (width_new, height_new)
    #Warping the left image with H to orient to stitch with right
    left_warped = cv2.warpPerspective(src=left, M=H, dsize=size)
    
    right_height, right_width, right_channels = right.shape
    height_new = int(round(abs(y) + right_height))
    width_new = int(round(abs(x) + right_width))
    size = (width_new, height_new)
    #Translating the right image to get it aligned with left image coordinate frame
    right_warped = cv2.warpPerspective(src=right, M=T_matrix, dsize=size)
    # Stitching procedure : Going through each and every pixels of the image shape. Take the left and right pixel
    # compare :
    #1. If left pixel is black , then take the intensity values of right and use it to modify our final image and vice versa.
    #2. If the both pixels are not black then combine the intensities of both pixels
    #In this manner two images get overlapped based on same features and the black pixels will be truncated
    
    if(left_warped.shape[1] < right_warped.shape[1]):
        final_shape = left_warped
    else:
        final_shape = right_warped
    
    for i in (range(final_shape.shape[0])):
        for j in range(final_shape.shape[1]):
            left_pixel = left_warped[i, j, :]
            right_pixel = right_warped[i, j, :]
            if (np.sum(left_pixel) > 0) and (int(np.sum(right_pixel)) == 0):
                left_warped[i, j, :] = left_pixel
            elif(int(np.sum(left_pixel))==0 ) and (np.sum(right_pixel) >0):
                left_warped[i, j, :] = right_pixel
            elif (np.sum(left_pixel) > 0) and (np.sum(right_pixel) >0):
                left_warped[i, j, :] = (left_pixel + right_pixel) / 2
            else:
                pass           
    stitch_image = left_warped[:right_warped.shape[0], :right_warped.shape[1], :]
    return stitch_image

def ransac(source_x,source_y,destination_x,destination_y):
    #Getting 4 random points .4 random point pairs are required since we construct homography on these 4 pairs with RANSAC

    max_inliers = 0             
    H_final = []
    threshold = 0.4
    iterations = 100

    for i in range(iterations):
        current_inliers = 0
        random_indices = np.random.choice(len(source_x),size =4)
        pa_x_rand = source_x[random_indices]
        pa_y_rand = source_y[random_indices]
        pb_x_rand = destination_x[random_indices]
        pb_y_rand = destination_y[random_indices]
        #computing homography matrix
        H = compute_homography(pa_x_rand,pa_y_rand,pb_x_rand,pb_y_rand)
        for i in range(len(source_x)):
            error = compute_error(H,source_x[i],source_y[i],destination_x[i],destination_y[i])
            if(error<threshold):
                current_inliers += 1
        if(current_inliers > max_inliers):
            max_inliers = current_inliers
            H_final = copy.deepcopy(H)

    return H_final

fig = plt.figure(figsize=(18,15))


sift_matches,source_x,source_y,destination_x,destination_y = sift_matcher(image1_gray,image2_gray,image1_rgb,image2_rgb)
H =  ransac(source_x,source_y,destination_x,destination_y)
print("Homography Matrix between Image 1 and Image 2")
print(H)
stitched_image12 = stitch_img(image1_rgb, image2_rgb, H)
ax = fig.add_subplot(321)
ax.imshow(sift_matches,cmap='gray')
ax = fig.add_subplot(322)
ax.imshow(stitched_image12)

sift_matches,source_x,source_y,destination_x,destination_y = sift_matcher(image3_gray,image4_gray,image3_rgb,image4_rgb)
H =  ransac(source_x,source_y,destination_x,destination_y)
stitched_image34 = stitch_img(image3_rgb, image4_rgb, H)
print("Homography Matrix between Image 3 and Image 4")
print(H)
ax = fig.add_subplot(323)
ax.imshow(sift_matches,cmap='gray')
ax = fig.add_subplot(324)
ax.imshow(stitched_image34)


sift_matches,source_x,source_y,destination_x,destination_y = sift_matcher(image2_gray,image3_gray,image2_rgb,image3_rgb)
H =  ransac(source_x,source_y,destination_x,destination_y)
print("Homography Matrix between Image 2 and Image 3")
final = stitch_img(stitched_image12, stitched_image34,H)
print(H)
ax = fig.add_subplot(325)
ax.imshow(sift_matches,cmap='gray')
ax = fig.add_subplot(326)
ax.imshow(final)


