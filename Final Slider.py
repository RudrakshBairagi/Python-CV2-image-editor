import cv2
import numpy as np

def p(x):
    pass

# User-defined function for the editor
def editor(a):
    img = cv2.imread(a)
    if img is None:
        print("Error: Could not load image")
        exit()
    
    cv2.namedWindow('Image')
    # Creating sliders for editing
    cv2.createTrackbar('R', 'Image', 0, 255, p)
    cv2.createTrackbar('G', 'Image', 0, 255, p)
    cv2.createTrackbar('B', 'Image', 0, 255, p)

    cv2.createTrackbar('Black', 'Image', 0, 255, p)
    cv2.createTrackbar('White', 'Image', 0, 255, p)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    initial_saturation = np.mean(hsv[:, :, 1])
    cv2.createTrackbar('Saturation', 'Image', int(initial_saturation), 255, p)

    cv2.createTrackbar('Sharpness', 'Image', 1, 100, p)
    cv2.createTrackbar('Reset','Image',0,1,p)
    switch = '0 : OFF\n1 : ON'
    cv2.createTrackbar(switch, 'Image', 0, 1, p)

    while True:
        # Cloning the original image
        copy = img.copy()
        # Getting values from sliders
        r = cv2.getTrackbarPos('R', 'Image')
        g = cv2.getTrackbarPos('G', 'Image')
        b = cv2.getTrackbarPos('B', 'Image')
        black = cv2.getTrackbarPos('Black', 'Image')
        white = cv2.getTrackbarPos('White', 'Image')
        sharpness = cv2.getTrackbarPos('Sharpness', 'Image')
        reset=cv2.getTrackbarPos('Reset', 'Image')
        saturation = cv2.getTrackbarPos('Saturation', 'Image')
        s = cv2.getTrackbarPos(switch, 'Image')

        if reset == 1:
        # Reset all trackbars to their default positions
         cv2.setTrackbarPos('R', 'Image', 0)
         cv2.setTrackbarPos('G', 'Image', 0)
         cv2.setTrackbarPos('B', 'Image', 0)
         cv2.setTrackbarPos('Black', 'Image', 0)
         cv2.setTrackbarPos('White', 'Image', 0)
         cv2.setTrackbarPos('Sharpness', 'Image', 1)
         cv2.setTrackbarPos('Saturation', 'Image', int(initial_saturation))
         cv2.setTrackbarPos('Reset', 'Image', 0)
        # Applying values
        if s == 0:
            copy[:] = 0
        else:
            copy[:, :, 0] = cv2.add(copy[:, :, 0], b)
            copy[:, :, 1] = cv2.add(copy[:, :, 1], g)
            copy[:, :, 2] = cv2.add(copy[:, :, 2], r)

            copy = cv2.addWeighted(copy, 1.0, copy, 0, white - black)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Base kernel for sharpening
            sharp_img = cv2.filter2D(copy, -1, kernel)
            copy = cv2.addWeighted(copy, 1.0, sharp_img, sharpness / 100, 0)

            hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.add(hsv[:, :, 1], saturation - int(initial_saturation))
            copy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
       
        h,w=copy.shape[:2]
        nw=w//2
        nh=h//2
        small= cv2.resize(copy,(nw,nh))
        
 

        cv2.imshow('Image', small)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press 'Esc' to exit
            break
        elif k == ord('s'):  # Press 's' to save the image
            cv2.imwrite('edited_image.jpg', copy)
            print("Image saved as 'edited_image.jpg'")

    cv2.destroyAllWindows()



editor('m4.jpg')
        
                
                
                
                
            
        
        
        
        
        
