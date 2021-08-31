import numpy as np
from numba import jit, prange
if __name__ !=  '__main__':
   import multiprocessing as mp
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import random

def mss_descriptor_gaussian(img):
    img = img.astype(np.dtype('f8'))
    shape = (img.shape[0],img.shape[1],4)
    #print("shpe is : ",shape)
    R = np.asarray([(0,1),(0,-1),(1,0),(-1,0)]) # espace de recherche
    R2 = np.asarray([(1,0),(-1,0),(0,-1),(0,1)])
    mind_img = np.zeros(shape,np.dtype('f8'))
    for i in range(2,len(img)-2):
        for j in range(2,len(img[i])-2):
            Dp = np.empty(4,np.dtype('f8'))
            for k,r in enumerate(R):
                l1 = (np.power(img[i+R2[k][0]-1][j+R2[k][1]-1]-img[i+r[0]-1][j+r[1]-1],2) + 2*np.power(img[i+R2[k][0]-1][j+R2[k][1]]-img[i+r[0]-1][j+r[1]],2) + np.power(img[i+R2[k][0]-1][j+R2[k][1]+1]-img[i+r[0]-1][j+r[1]+1],2))
                l2 = (2*np.power(img[i+R2[k][0]][j+R2[k][1]-1]-img[i+r[0]][j+r[1]-1],2) + 4*np.power(img[i+R2[k][0]][j+R2[k][1]]-img[i+r[0]][j+r[1]],2) + 2*np.power(img[i+R2[k][0]][j+R2[k][1]+1]-img[i+r[0]][j+r[1]+1],2))
                l3 = (np.power(img[i+R2[k][0]+1][j+R2[k][1]-1]-img[i+r[0]+1][j+r[1]-1],2) + 2*np.power(img[i+R2[k][0]+1][j+R2[k][1]]-img[i+r[0]+1][j+r[1]],2) + np.power(img[i+R2[k][0]+1][j+R2[k][1]+1]-img[i+r[0]+1][j+r[1]+1],2))
                Dp[k] = (1/16)*(l1+l2+l3)
            V = np.sum(Dp)/4.0 + 0.000001
            mind_img[i][j][0] = np.exp(-1*(Dp[0]/V))
            mind_img[i][j][1] = np.exp(-1*(Dp[1]/V))
            mind_img[i][j][2] = np.exp(-1*(Dp[2]/V))
            mind_img[i][j][3] = np.exp(-1*(Dp[3]/V))
    return mind_img




def mind_descriptor_gaussian(img):
    img = img.astype(np.dtype('f8'))
    shape = (img.shape[0],img.shape[1],4)
    #print("shpe is : ",shape)
    R = np.asarray([(0,1),(0,-1),(1,0),(-1,0)]) # espace de recherche
    mind_img = np.zeros(shape,np.dtype('f8'))
    for i in range(2,len(img)-2):
        for j in range(2,len(img[i])-2):
            Dp = np.empty(4,np.dtype('f8'))
            patch_p = [[img[i-1][j-1],img[i-1][j],img[i-1][j+1]],[img[i][j-1],img[i][j],img[i][j+1]],[img[i+1][j-1],img[i+1][j],img[i+1][j+1]]]
            for k,r in enumerate(R):
                l1 = (np.power(patch_p[0][0]-img[i+r[0]-1][j+r[1]-1],2) + 2*np.power(patch_p[0][1]-img[i+r[0]-1][j+r[1]],2) + np.power(patch_p[0][2]-img[i+r[0]-1][j+r[1]+1],2))
                l2 = (2*np.power(patch_p[1][0]-img[i+r[0]][j+r[1]-1],2) + 4*np.power(patch_p[1][1]-img[i+r[0]][j+r[1]],2) + 2*np.power(patch_p[1][2]-img[i+r[0]][j+r[1]+1],2))
                l3 = (np.power(patch_p[2][0]-img[i+r[0]+1][j+r[1]-1],2) + 2*np.power(patch_p[2][1]-img[i+r[0]+1][j+r[1]],2) + np.power(patch_p[2][2]-img[i+r[0]+1][j+r[1]+1],2))
                Dp[k] = (1/16)*(l1+l2+l3)
            V = np.sum(Dp)/4.0 + 0.000001
            mind_img[i][j][0] = np.exp(-1*(Dp[0]/V))
            mind_img[i][j][1] = np.exp(-1*(Dp[1]/V))
            mind_img[i][j][2] = np.exp(-1*(Dp[2]/V))
            mind_img[i][j][3] = np.exp(-1*(Dp[3]/V))
    return mind_img

# @parfor(range(10), (3,))
# def fun(i, a):
#     sleep(1)
#     return a*i**2

def call_inner(i,img,mind_img,R):
    pool2 = mp.Pool()
    result2 =pool2.map(inner_workings, i, range(2,len(img[i])-2),img,mind_img,R)
    return result2

def inner_workings(i,j,img,mind_img,R):
    Dp = np.empty(4,np.dtype('f8'))
    patch_p = [[img[i-1][j-1],img[i-1][j],img[i-1][j+1]],[img[i][j-1],img[i][j],img[i][j+1]],[img[i+1][j-1],img[i+1][j],img[i+1][j+1]]]
    for k,r in enumerate(R):
        l1 = (np.power(patch_p[0][0]-img[i+r[0]-1][j+r[1]-1],2) + 2*np.power(patch_p[0][1]-img[i+r[0]-1][j+r[1]],2) + np.power(patch_p[0][2]-img[i+r[0]-1][j+r[1]+1],2))
        l2 = (2*np.power(patch_p[1][0]-img[i+r[0]][j+r[1]-1],2) + 4*np.power(patch_p[1][1]-img[i+r[0]][j+r[1]],2) + 2*np.power(patch_p[1][2]-img[i+r[0]][j+r[1]+1],2))
        l3 = (np.power(patch_p[2][0]-img[i+r[0]+1][j+r[1]-1],2) + 2*np.power(patch_p[2][1]-img[i+r[0]+1][j+r[1]],2) + np.power(patch_p[2][2]-img[i+r[0]+1][j+r[1]+1],2))
        Dp[k] = (1/16)*(l1+l2+l3)
    V = np.sum(Dp)/4.0 + 0.000001
    mind_img[i][j][0] = np.exp(-1*(Dp[0]/V))
    mind_img[i][j][1] = np.exp(-1*(Dp[1]/V))
    mind_img[i][j][2] = np.exp(-1*(Dp[2]/V))
    mind_img[i][j][3] = np.exp(-1*(Dp[3]/V))
  

def mind_descriptor_gaussian2nojit(img):

    img = img.astype(np.dtype('f8'))
    shape = (img.shape[0],img.shape[1],4)
    #print("shpe is : ",shape)
    R = np.asarray([(0,1),(0,-1),(1,0),(-1,0)]) # espace de recherche
    mind_img = np.zeros(shape,np.dtype('f8'))
    
    pool1 = mp.Pool(mp.cpu_count())
    pool1.apply(inner_workings, args=(range(2,len(img)-2), range(2,len(img[0])-2) ,img,mind_img,R))
    pool1.close()
    return mind_img


@jit
def inner_workings(i,j,img,mind_img,R):
    Dp = np.empty(4,np.dtype('f8'))
    patch_p = [[img[i-1][j-1],img[i-1][j],img[i-1][j+1]],[img[i][j-1],img[i][j],img[i][j+1]],[img[i+1][j-1],img[i+1][j],img[i+1][j+1]]]
    for k,r in enumerate(R):
        l1 = (np.power(patch_p[0][0]-img[i+r[0]-1][j+r[1]-1],2) + 2*np.power(patch_p[0][1]-img[i+r[0]-1][j+r[1]],2) + np.power(patch_p[0][2]-img[i+r[0]-1][j+r[1]+1],2))
        l2 = (2*np.power(patch_p[1][0]-img[i+r[0]][j+r[1]-1],2) + 4*np.power(patch_p[1][1]-img[i+r[0]][j+r[1]],2) + 2*np.power(patch_p[1][2]-img[i+r[0]][j+r[1]+1],2))
        l3 = (np.power(patch_p[2][0]-img[i+r[0]+1][j+r[1]-1],2) + 2*np.power(patch_p[2][1]-img[i+r[0]+1][j+r[1]],2) + np.power(patch_p[2][2]-img[i+r[0]+1][j+r[1]+1],2))
        Dp[k] = (1/16)*(l1+l2+l3)
    V = np.sum(Dp)/4.0 + 0.000001
    mind_img[i][j][0] = np.exp(-1*(Dp[0]/V))
    mind_img[i][j][1] = np.exp(-1*(Dp[1]/V))
    mind_img[i][j][2] = np.exp(-1*(Dp[2]/V))
    mind_img[i][j][3] = np.exp(-1*(Dp[3]/V))
  
@jit
def mind_descriptor_gaussian2(img):

    img = img.astype(np.dtype('f8'))
    shape = (img.shape[0],img.shape[1],4)
    #print("shpe is : ",shape)
    R = np.asarray([(0,1),(0,-1),(1,0),(-1,0)]) # espace de recherche
    mind_img = np.zeros(shape,np.dtype('f8'))
    lesI = range(2,len(img)-2)
    lesJ = range(2,len(img[0])-2)
    for i in prange(2, len(img)-2):
        for j in prange(2,len(img[0])-2):
            inner_workings(i, j, img, mind_img, R)
    return mind_img

@jit
def inner_workings2(i,j,img,mind_img,R):
    Dp = np.empty(len(R),np.dtype('f8'))
    patch_p = [[img[i-1][j-1],img[i-1][j],img[i-1][j+1]],[img[i][j-1],img[i][j],img[i][j+1]],[img[i+1][j-1],img[i+1][j],img[i+1][j+1]]]
    for k,r in enumerate(R):
        l1 = (np.power(patch_p[0][0]-img[i+r[0]-1][j+r[1]-1],2) + 2*np.power(patch_p[0][1]-img[i+r[0]-1][j+r[1]],2) + np.power(patch_p[0][2]-img[i+r[0]-1][j+r[1]+1],2))
        l2 = (2*np.power(patch_p[1][0]-img[i+r[0]][j+r[1]-1],2) + 4*np.power(patch_p[1][1]-img[i+r[0]][j+r[1]],2) + 2*np.power(patch_p[1][2]-img[i+r[0]][j+r[1]+1],2))
        l3 = (np.power(patch_p[2][0]-img[i+r[0]+1][j+r[1]-1],2) + 2*np.power(patch_p[2][1]-img[i+r[0]+1][j+r[1]],2) + np.power(patch_p[2][2]-img[i+r[0]+1][j+r[1]+1],2))
        Dp[k] = (1/16)*(l1+l2+l3)
    V = np.sum(Dp)/8.0 + 0.000001
    for l in range(len(R)):
        mind_img[i][j][l] = np.exp(-1*(Dp[l]/V))
    # mind_img[i][j][0] = np.exp(-1*(Dp[0]/V))
    # mind_img[i][j][1] = np.exp(-1*(Dp[1]/V))
    # mind_img[i][j][2] = np.exp(-1*(Dp[2]/V))
    # mind_img[i][j][3] = np.exp(-1*(Dp[3]/V))
    # mind_img[i][j][4] = np.exp(-1*(Dp[4]/V))
    # mind_img[i][j][5] = np.exp(-1*(Dp[5]/V))
    # mind_img[i][j][6] = np.exp(-1*(Dp[6]/V))
    # mind_img[i][j][7] = np.exp(-1*(Dp[7]/V))
  
@jit
def mind_descriptor_gaussian3(img):

    img = img.astype(np.dtype('f8'))
    shape = (img.shape[0],img.shape[1],8)
    R = np.asarray([(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(-1,1),(1,1),(1,-1)]) # espace de recherche
    mind_img = np.zeros(shape,np.dtype('f8'))
    lesI = range(2,len(img)-2)
    lesJ = range(2,len(img[0])-2)
    for i in prange(2, len(img)-2):
        for j in prange(2,len(img[0])-2):
            inner_workings2(i, j, img, mind_img, R)
    return mind_img


def PSNR(img1, img2):
    MAX = 1
    mind1 = mind_descriptor_gaussian3(img1)
    mind2 = mind_descriptor_gaussian3(img2)
    # plt.subplot(121), plt.imshow(mind1[:,:,3])
    # plt.subplot(122), plt.imshow(mind2[:,:,3]), plt.show()
    mse = 0.0
    for channel in range(mind1.shape[2]):
        for i in range(mind1.shape[0]):
            for j in range(mind1.shape[1]):
                mse += np.power(mind1[i,j,channel]-mind2[i,j,channel], 2)
    dividend = mind1.shape[0]*mind1.shape[1]*mind1.shape[2]
    msed = mse/dividend
    psnr = 20*np.log10(MAX) - 10*np.log10(msed)
    return psnr, mse

# Calculates PSNR from the mind calculation of registred images
def PSNRh(img1, img2, homography, show = False):
    if homography.any() != None:
        po1 = np.matmul(np.asarray(homography),np.asarray([0.0, 0.0, 1]))
        po2 = np.matmul(np.asarray(homography),np.asarray([0.0, img2.shape[0], 1]))
        po3 = np.matmul(np.asarray(homography),np.asarray([img2.shape[1], img2.shape[0], 1]))
        po4 = np.matmul(np.asarray(homography),np.asarray([img2.shape[1], 0.0, 1]))
        po1/= po1[2] 
        po2/= po2[2]
        po3/= po3[2]
        po4/= po4[2]
        d1= [ int(max(po1[1],po4[1])),int(max(po1[0],po2[0]))]
        d2= [ int(min(po2[1],po3[1])),int(min(po3[0],po4[0]))]
        if(d1[0] > img1.shape[0]): d1[0] = 0
        if(d1[1] > img1.shape[1]): d1[1] = 0
        if(d2[0] > img1.shape[0]): d2[0] = img1.shape[0]
        if(d2[1] > img1.shape[1]): d2[1] = img1.shape[1]
        img1 = img1[d1[0]:d2[0],d1[1]:d2[1]]
        img2 = img2[d1[0]:d2[0],d1[1]:d2[1]]
        if( show == True):
            plt.subplot(121), plt.imshow(img1)
            plt.subplot(122), plt.imshow(img2), plt.show()
            
    MAX = 1
    mind1 = mind_descriptor_gaussian3(img1)
    mind2 = mind_descriptor_gaussian3(img2)
    
    mse = 0.0
    for channel in range(mind1.shape[2]):
        for i in range(mind1.shape[0]):
            for j in range(mind1.shape[1]):
                mse += np.power(mind1[i,j,channel]-mind2[i,j,channel], 2)
    dividend = mind1.shape[0]*mind1.shape[1]*mind1.shape[2]
    msed = mse/dividend
    psnr = 20*np.log10(MAX) - 10*np.log10(msed)
    return psnr, mse, img1, img2
    

# Performs the registration on MIND images and then calculates the PSNR
def PSNR2(mind1,mind2, homography):
    MAX = 255
    mind1 = 255*mind_descriptor_gaussian3(img1)
    mind2 = 255*mind_descriptor_gaussian3(img2)
    mse = 0.0
    



def recaleAutoMind(img1, img2, show = False, iterate = 0):
    font = cv.FONT_HERSHEY_SIMPLEX
    blockSize = 2
    apertureSize = 3
    k = 0.04
    kp = []
    bestMatch = 0.0
    bestHomo = []
    bestWarp = img2
    kernel = np.asarray([[-1/9,-1/9,-1/9],[-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]]) *9
    dst = cv.cornerHarris( img1, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    thresh = np.floor(np.mean(dst_norm_scaled))
    thresh1 = np.median(dst_norm)
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if (dst_norm[i,j] < thresh1 -0.001) or (dst_norm[i,j] > thresh1 + 0.001):
                kp.append(cv.KeyPoint(j,i,10))
    print("nb points vis= ", len(kp), img1.shape[0]*img1.shape[1])
    kp2 = []
    # Detecting corners
    dst2 = cv.cornerHarris(img2, blockSize, apertureSize, k)
    # Normalizing
    dst_norm2 = np.empty(dst2.shape, dtype=np.float32)
    cv.normalize(dst2, dst_norm2, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled2 = cv.convertScaleAbs(dst_norm2)
    thresh22 = np.median(dst_norm2)
    thresh2 = np.floor(np.mean(dst_norm_scaled2))
    print("thresh = ", thresh22)
    for i in range(dst_norm2.shape[0]):
        for j in range(dst_norm2.shape[1]):
            if (dst_norm2[i,j] < thresh22-0.001) or (dst_norm2[i,j] > thresh22+0.001) :
                kp2.append(cv.KeyPoint(j,i,10))   
    print("nb points IR= ", len(kp2), img2.shape[0]*img2.shape[1])
    
    # fast = cv.FastFeatureDetector_create()
    # kp2 = fast.detect(img2,None)
    
    mind1 = mind_descriptor_gaussian3(img1)
    mind2 = mind_descriptor_gaussian3(img2)
    points1 = img1
    points2 = img2
    if(show == True):
        points1 = cv.drawKeypoints(img1, kp, points1)
        points2 = cv.drawKeypoints(img2, kp2, points2)
        plt.subplot(121), plt.imshow(points1)
        plt.subplot(122), plt.imshow(points2), plt.show()
    
    sift = cv.SIFT_create()
    pts1 = []
    pts2 = []
    # orb = cv.xfeatures2d.BriefDescriptorExtractor_create()
    for channel in range(mind1.shape[2]):
        kpo, des = sift.compute((255*mind1[:,:,channel]).astype(np.uint8), kp)
        kpo2, des2 = sift.compute((255*mind2[:,:,channel]).astype(np.uint8), kp2)
        matcher = cv.BFMatcher_create(cv.NORM_L2 ,crossCheck=True)
        matches = matcher.match(des,des2)
        samples = []
        try:
            samples = random.sample(matches, 13)
        except:
            print("Sample larger than population, Number of matches = ", len(matches))
        if(show == True):
            points1 = cv.drawMatches(img1, kpo, img2, kpo2, samples, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            for samp in samples:
                coord = (int(kpo[samp.queryIdx].pt[0]),int(kpo[samp.queryIdx].pt[1]))
                # cv.putText(points1, str(int(samp.distance)), coord, font, 0.3, (255, 255, 255), 1, cv.LINE_AA)
            plt.imshow(points1), plt.show()
        
        matches = sorted(matches, key = lambda x:x.distance)
        print("len : ", len(matches), "min dist : ", min(matches, key = lambda x:x.distance) )
        #if(len(matches) > 10) : matches = matches[:int(len(matches)]
        print("len2 : ", len(matches))
        for j, match in enumerate(matches):
                pts1.append(kpo[match.queryIdx].pt)
                pts2.append(kpo2[match.trainIdx].pt)
        
        
        print(len(pts1))
        nppts1 = np.asarray(pts1)
        nppts2 = np.asarray(pts2)
        try:
            homography, mask = cv.findHomography(nppts2 ,nppts1,cv.RANSAC)
            warped = cv.warpPerspective(img2,homography, (img1.shape[1], img1.shape[0]))
        except:
            continue
        
        bestZone1 = img1
        bestZOne2 = img2
        try:
            scoreH, mseScoreH, zone1, zone2 = PSNRh(img1, warped, homography, show)
        except:
            scoreH = 1
        scoreOri, mseScoreOri = PSNR(img1,img2)
        if bestMatch < scoreH :
            bestMatch = scoreH
            bestHomo = homography
            bestWarp = warped
        try:
            bestZone1 = zone1
            bestZone2 = zone2
        except:
            continue
        if(show == True):
            plt.subplot(221), plt.imshow(img1, "gray")
            plt.subplot(222), plt.imshow(img2, "gray"), 
            plt.subplot(223), plt.imshow(warped, "gray"), 
            plt.subplot(224), plt.imshow(img1, "gray", alpha = 0.5), plt.imshow(warped, "gray", alpha = 0.5), 
            # plt.suptitle("PSNR : "+str(scoreH)+" - "+str(scoreOri)+'\n'+"MSE : "+str(mseScoreH)+" - "+str(mseScoreOri)), plt.show()
            plt.suptitle("Gain PSNR : "+str(scoreH - scoreOri)+"Db"), plt.show()
    if(iterate > 0 ): return recaleAutoMind(bestZone1, bestZone2, show , iterate-1)
    else : return bestWarp, bestHomo, bestMatch, dst_norm2
        
############## TESTING AREA - DO NOT FORGET TO COMMENT AREA AFTER TESTS #####################

truc = "calib2\VIS\*.png"
VIs = glob.glob("calib3\VIS\*.jpg")
IRs = glob.glob("calib3\IR\*.png")

img = cv.imread(VIs[24],0)
img2 = cv.imread(IRs[24],0)

img1 = cv.resize(img, (160,120), cv.INTER_AREA)
img2 = cv.rotate(img2, cv.ROTATE_180)


# path1 = "dataset1/"
# img = cv.imread(path1+"doigt2-visible-04.jpg")   # les images vont de 1 a 113 et de A-1 a A-8 et de E-1 a E-5
# img2 = cv.imread(path1+"doigt2-Thermal-04.raw.png")
# img = cv.flip(img, 0)
# img = cv.flip(img, 1)
# img = cv.resize(img, (img2.shape[1], img2.shape[0]), interpolation=cv.INTER_AREA)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #TUFTS est en BGR de base
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)



img = cv.imread("dataset1\\rond_noir.png")
img1 = cv.imread("dataset1\\rond_noir.png",0)
img2 = cv.imread("dataset1\\rond_blanc.png",0)

# rgbpath = "tuftsRGB/"
# thermalpath = "tuftsThermal/"
# img = cv.imread(rgbpath+"63-TD-A-1.jpg")
# img2 = cv.imread(thermalpath+"63-TD-A-1.jpg")
# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# path1 = "dataset2\lab\\take5\\2"
# # path1 = "dataset2\Camouflage\\sync and unreg\\take_1"
# path1 = "dataset2\Patio\\take_1"
# img = cv.imread(path1+"\VIS\VIS_1600.jpg")   # les images vont de 1 a 113 et de A-1 a A-8 et de E-1 a E-5
# img2 = cv.imread(path1+"\IR\IR_1600.jpg")
# img = cv.resize(img, (img.shape[1]//3, img.shape[0]//3), interpolation=cv.INTER_AREA)
# img2 = cv.resize(img2, (img2.shape[1]//3, img2.shape[0]//3), interpolation=cv.INTER_AREA)
# img1 = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #TUFTS est en BGR de base
# img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

warp, homo, score, dst = recaleAutoMind(img1, img2, True,0)

plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(img2, cmap="gray"), 
plt.subplot(223), plt.imshow(warp, cmap="gray"), 
plt.subplot(224), plt.imshow(img1, alpha = 0.5), plt.imshow(warp, alpha = 0.5, cmap="gray"), 
plt.suptitle("PSNR : "+str(score) )
plt.show()