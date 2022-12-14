import cv2 as cv 
import numpy as np 

def image_show(path,winname = 'oepncv_learn'):
    image = cv.imread(path)
    cv.imshow(winname,image)
    cv.waitKey(0)

def color_space_transfor(path):
    img = cv.imread(path)
    cv.imshow('BGR',img)
    hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    ycrcb_img = cv.cvtColor(img,cv.COLOR_BGR2YCrCb)
    cv.imshow('HSV',hsv_img)
    cv.imshow('YCRCB',ycrcb_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def image_create():
    img = np.zeros((3,3,3),dtype=np.uint8)
    cv.imshow('black',img)
    img[:] = 255
    cv.imshow('white',img)
    img[:] = (255,0,0)
    cv.imshow('blue',img)
    # cv.resizeWindow('black',200,200)
    cv.waitKey(0)
    cv.destroyAllWindows()

def image_red_blue():
    img = np.zeros((512,512,3),dtype=np.uint8)
    cv.imshow('black',img)
    img[:,:256] = (0,0,255)
    img[:,256:] = (255,0,0)
    cv.imshow('red_blue',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def image_inverse(path):
    img = cv.imread(path)
    cv.imshow('normal',img)
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            b, g, r = img[i,j]
            print(i,b,g,r)
            img[i,j] = (255 - b, 255 - g, 255 - r)
    cv.imshow('inverse',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def image_calculate(path):
    img = cv.imread(path)
    cv.imshow('normal',img)
    img1 = np.zeros_like(img)
    mask = np.zeros_like(img)
    mask[1000:3000,2000:4000] = 1
    dst = cv.add(img,img1,mask=mask)
    cv.imshow('add_mask',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    file = '../data/beauty.png'
    # image_show(file,'beauty')
    # color_space_transfor(file)
    # image_create()
    # image_red_blue() 
    # image_inverse(file)
    image_calculate(file)

if __name__ == "__main__":
    main()
