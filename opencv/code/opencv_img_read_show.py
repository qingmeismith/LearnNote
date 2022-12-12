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


def main():
    file = '../data/beauty.png'
    # image_show(file,'beauty')
    # color_space_transfor(file)
    # image_create()
    image_red_blue()

if __name__ == "__main__":
    main()
