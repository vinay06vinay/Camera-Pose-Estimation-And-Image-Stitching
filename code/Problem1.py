import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as RB

video_object = cv2.VideoCapture("project2.avi")
x_center = []
y_center = []
#Lists containing translation plots
norm_trans_list = []
x_frame_count = []
pitch_list=[]
roll_list=[]
yaw_list=[]
xdisp = []
ydisp = []
zdisp = []
if (video_object.isOpened == False):
    print("Error Streaming the video")
def compute_homography(corners,world_corners):
    A=[]
    for i in range(len(corners)):
        src_x = world_corners[i][0]
        src_y = world_corners[i][1]
        dest_x = corners[i][0]
        dest_y = corners[i][1]
        row1 = [src_x,src_y,1,0,0,0,-dest_x*src_x,-dest_x*src_y,-dest_x]
        row2 = [0,0,0,src_x,src_y,1,-dest_y*src_x,-dest_y*src_y,-dest_y]
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
def decompose_homography(H):
    # print(H.shape)
    K = np.array([[1.38e+03, 0, 9.46e+02],
            [0, 1.38e+03, 5.27e+02],
            [0, 0, 1]]) 
    K[:2,:] /= 3
    # Compute inverse of K
    K_inv = np.linalg.inv(K)
    B = K_inv @ H
    # Compute lambda
    l = np.linalg.norm(B[:,:2]) / np.sqrt(2)

    # Compute r1, r2, r3
    r1 = B[:,0] / l
    r2 = B[:,1] / l
    r3 = np.cross(r1, r2)
    # Compute t
    t = B[:,2] / l
    
    # Compute R
    R = np.column_stack((r1, r2, r3))
    return R, t   
def compute_hough_lines(edges,frame):

    #threshold taking as 120 in the count in accumulator array so that maximum number of lines passing through
    #point in hough space
    thresh = 20
    #Taking the value of theta in hough space from 0-180
    theta = np.arange(0,181,1)
    #rho is the diagonal length of the image
    width,height = edges.shape[1],edges.shape[0]
    rho_diagonal = round(np.sqrt(np.square(width)+np.square(height)))
    #Initialisation of accumulator array with zero values
    accumulator = np.zeros((2 * rho_diagonal, len(theta)), dtype=np.uint8)
    y_edge,x_edge = np.where(edges ==255)
    for i in range(len(x_edge)):
        x = x_edge[i]
        y = y_edge[i]
        for j in range(len(theta)):
            rho = int(round((x*np.cos(np.deg2rad(theta[j]))) + (y*np.sin(np.deg2rad(theta[j]))) ) )
            accumulator[rho,j] += 3
    #accumulator edge pixel in hough space satisfying the threshold condition
    acc_edge = np.where(accumulator > thresh)
    thetas = list(acc_edge[1])
    rhos = list(acc_edge[0])
    dict_unique = {}
    for i in range(len(thetas)):
        if(rhos[i] not in dict_unique):
            dict_unique[rhos[i]] = [(rhos[i],thetas[i])]
        else:
            dict_unique[rhos[i]].append((rhos[i],thetas[i]))
    dict_final = {}
    for key,value in dict_unique.items():
        if(len(value) >  12 ):
            dict_final[key] = value
    # print(dict_final)
            
        
            
    thetas = [k[1] for j,i in dict_final.items() for k in i]
    rhos   = [k[0] for j,i in dict_final.items() for k in i]
    x_a = []
    y_a = []
    x_b = []
    y_b = []
    x_c = []
    y_c = []
    x_d = []
    y_d = []
    for i in range(0, len(thetas)):
        a = np.cos(np.deg2rad(thetas[i]))
        b = np.sin(np.deg2rad(thetas[i]))
        x0 = a*rhos[i]
        y0 = b*rhos[i]
        # print(x0,y0)
        if((x0>750 and x0<850) and (y0>440 and y0 < 490)):
            x_a.append(x0)
            y_a.append(y0)
        elif((x0 > 850 and x0 < 980) and (y0>180 and y0<290)):
            x_b.append(x0)
            y_b.append(y0)
        elif((x0>1150 and x0 < 1250) and (y0>400 and y0<415)):
            x_c.append(x0)
            y_c.append(y0)
        elif((x0>1040 and x0 < 1140) and (y0>590and y0<640)):
            x_d.append(x0)
            y_d.append(y0)
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
        
    if(len(x_a) == 0 or len(x_b) == 0 or len(x_c) == 0 or len(x_d) == 0):
        return

    corners = [(sum(x_a)/len(x_a),sum(y_a)/len(y_a)),(sum(x_b)/len(x_b),sum(y_b)/len(y_b)),(sum(x_c)/len(x_c),sum(y_c)/len(y_c)),(sum(x_d)/len(x_d),sum(y_d)/len(y_d))]
    return corners



    
count=0
while (video_object.isOpened):
    ret, frame = video_object.read()
    if ret == True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        low_range = np.array([60,10,230],np.uint8)
        high_range = np.array([200,255,255],np.uint8)
        mask= cv2.inRange(hsv,low_range,high_range)
        blur = cv2.GaussianBlur(mask, (15, 15),cv2.BORDER_DEFAULT)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.Canny(blur,120,200)
        corners = compute_hough_lines(edges,frame)
        if(corners):
            for i in corners:
                cv2.circle(frame,(int(i[0]),int(i[1])),3,(255,0,255),-1)
            H = compute_homography(corners,world_corners)
            R, t = decompose_homography(H)
            euler_angles = RB.from_matrix(R).as_euler('zxy', degrees=True)
            norm_trans_list.append(np.linalg.norm(t))
            roll_list.append(euler_angles[0])
            pitch_list.append(euler_angles[1])
            yaw_list.append(euler_angles[2])
            x_frame_count.append(count)
            xdisp.append(t[0])
            ydisp.append(t[1])
            zdisp.append(t[2])
            cv2.imshow("Edges",edges)  
            cv2.imshow("Frame",frame)
        count+=1
        if( count== 30):
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
print("Sample Homography Matrix in last iteration")
print(H)

video_object.release()
cv2.destroyAllWindows()
plt.plot(x_frame_count,norm_trans_list)
plt.xlabel("Number of Frames")
plt.ylabel("Translation")
plt.title("Translation plot of the camera")
plt.show()
plt.plot(x_frame_count,roll_list,label="Roll angles")
plt.plot(x_frame_count,pitch_list,label="Pitch angles")
plt.plot(x_frame_count,yaw_list,label="Yaw angles")
plt.xlabel("Number of Frames")
plt.ylabel("RPY in degrees")
plt.title("Roll pitch yaw plot")
plt.legend()
plt.show()


plt.plot(x_frame_count,xdisp,label="x variation")
plt.plot(x_frame_count,ydisp,label="y variation")
plt.plot(x_frame_count,zdisp,label="z variation")
plt.xlabel("Number of Frames")
plt.ylabel("X, Y and Z variation in meters")
plt.title("XYZ plot")
plt.legend()
plt.show()
