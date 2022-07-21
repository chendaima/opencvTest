from turtle import distance
from unittest import result
import cv2
from cv2 import Mat
import numpy as np
import math
#透射变换
def TousheTramfor():
    im_src = cv2.imread('images/build.jpeg')
    h, w, c = im_src.shape

    # 原始图像中物体的四个顶点的信息
    pts_src = np.array([(0, 0), (640, 0), (0, 520), (640, 520)], dtype="float32")
    # 目标物体中的物体的四个顶点信息
    pts_dst = np.array([(265, 30), (796, 99), (100, 473), (932, 373)], dtype="float32")

    # 是一个3x3的矩阵，根据对应的两个点，计算出变换矩阵，由此将原始图像进行转换。
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    print(M.shape)
    print(M)

    # 基于单应性矩阵，将原始图像转换成目标图像
    im_out = cv2.warpPerspective(im_src, M, (w, h))

    # plt.figure()
    # plt.subplot(1, 2, 1), plt.imshow(im_src[:, :, ::-1]), plt.title('src')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(im_out[:, :, ::-1]), plt.title('out')
    # plt.xticks([]), plt.yticks([])

    im_out.show()  # show dst

def motion_blur(img, degree=2, angle=1):
    image = img.copy()
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
    #读取原始图像，获取原始图像的行和列信息，设置中心点及光照强度参数，新建目标图像
def GetImage():
    img=cv2.imread('D:\\imageTestSar.png')#E:\\微信截图_20220717214654.png
#     rows,cols,chn=img.shape
#     #设置中心点
#     centerX=546#rows/2
#     centerY=910#cols/2

#     radius=100#min(centerX,centerY)#半径
#     #设置光照强度
#     strength=100
#     #新建目标图像
#     dst=np.zeros((rows,cols,3),dtype="uint8")

#    #添加图像光照特效
#     for i in range(rows):
#         for j in range(cols):
#             #计算当前点到光照中心的距离（平面两点之间的距离）
#             distance=math.pow((centerY-j),2)+math.pow((centerX-i),2)
#             #获取原始图像
#             B=img[i,j][0]
#             G=img[i,j][1]
#             R=img[i,j][2]
#             if(distance<radius*radius):
#                 #按照距离大小计算增强的光照值
#                 result=(int)(strength*(1.0-math.sqrt(distance)/radius))
#                 B=img[i,j][0]+result
#                 G=img[i,j][1]+result
#                 R=img[i,j][2]+result
#                 B=min(255,max(0,B))
#                 G=min(255,max(0,G))
#                 R=min(255,max(0,R))
#                 dst[i,j]=np.uint8((B,G,R))
#             else:
#                 dst[i,j]=np.uint8((B,G,R))

    #cv2.imshow('111',dst)
    #cv2.waitKey()
    blurred = motion_blur(img)
   # blurred=dst
    # for k in range(-50,50):
    #     B1=img[centerX,centerY+k][0]=255-(-k*3 if k<0 else k*3) 
    #     G1=img[centerX,centerY+k][1]=255-(-k*3 if k<0 else k*3) 
    #     R1=img[centerX,centerY+k][2]=255-(-k*3 if k<0 else k*3) 
    #     B1=min(255,max(0,B1))
    #     G1=min(255,max(0,G1))
    #     R1=min(255,max(0,R1))
    #     blurred[centerX,centerY+k]=np.uint8((B1,G1,R1))
    # for g in range(-50,50):
    #     B11=img[centerX+g,centerY][0]=255-(-g*3 if g<0 else g*3) 
    #     G11=img[centerX+g,centerY][1]=255-(-g*3 if g<0 else g*3) 
    #     R11=img[centerX+g,centerY][2]=255-(-g*3 if g<0 else g*3) 
    #     B11=min(255,max(0,B11))
    #     G11=min(255,max(0,G11))
    #     R11=min(255,max(0,R11))
    #     blurred[centerX+g,centerY]=np.uint8((B11,G11,R11))
    cv2.imwrite("D:\\imageTestSar.png",blurred)
    cv2.imshow('LightAndBlur',blurred)
   
    cv2.waitKey()
if __name__ == '__main__':
    GetImage()
    
   
   
    #cv2.imwrite(r'D:\\XTDJ_image\\XTDJ_f16_800_500_350_13_0_1_1.jpg', blurred)
