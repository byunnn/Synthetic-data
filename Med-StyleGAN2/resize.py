import cv2
import os


dirpath_img = '/home/ramsey/repos/x-rayGAN/data/x-ray/normal'


list_img_names = os.listdir(dirpath_img)

dirpath_target = '/home/ramsey/repos/x-rayGAN/data/med/train/normal'

for f_img in list_img_names:
    print(f_img)
          
    image_path = dirpath_img + '/' + f_img
    oimage = cv2.imread(image_path)

    new_shape = (512, 512)
    print('new_shape :', new_shape)
    resized_img = cv2.resize(oimage, new_shape)
    img_path = os.path.join(dirpath_target, f_img )

    cv2.imwrite(img_path,resized_img)
    cv2.waitKey(0)
