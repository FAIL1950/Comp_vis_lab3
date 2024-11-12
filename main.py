import cv2
from function import convert_to_grayscale, binarizationWithThreshold, OtsuBinarization, XDoGBinarization, XDoG, SRG

if __name__ == '__main__':
    try:
        img = convert_to_grayscale("img.jpg")
        img2 = convert_to_grayscale("img2.png")
        #res = binarizationWithThreshold(img, threshold=100)
        #res = OtsuBinarization(img, 162)
        #res = XDoGBinarization(img)
        #res = XDoG(img)
        res = SRG(img2)
        if res is not None:
            cv2.imshow("Input", img)
            cv2.imshow("Res", res)
            cv2.waitKey(0)
    except ValueError as e:
        print(e)



