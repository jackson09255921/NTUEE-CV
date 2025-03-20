import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        image_copy = image.copy()

        for octave in range(self.num_octaves):
            octave_images = []
            octave_images.append(image_copy)
            for i in range(1, self.num_guassian_images_per_octave):
                blurred = cv2.GaussianBlur(image_copy, ksize=(0,0), sigmaX=self.sigma ** i, sigmaY = self.sigma ** i)
                octave_images.append(blurred)
            
            gaussian_images.append(octave_images)
            last_img = octave_images[-1]
            image_copy = cv2.resize(last_img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []

        for octave in range(self.num_octaves):
            dog_images_in_octaves = []
            for i in range(self.num_DoG_images_per_octave):
                subtract = cv2.subtract(gaussian_images[octave][i + 1], gaussian_images[octave][i])
                dog_images_in_octaves.append(subtract)

                # Normalize and convert to uint8 for correct saving
                # subtract = cv2.normalize(subtract, None, 0, 255, cv2.NORM_MINMAX)
                # subtract = np.uint8(subtract)

                # file_name = f"DoG{octave+1}-{i+1}.png"
                # path = f"./image/{file_name}"
                # cv2.imwrite(path, subtract)


            dog_images.append(dog_images_in_octaves)
    

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = []

        for index in range(self.num_octaves):
            for i in range(1, self.num_DoG_images_per_octave - 1):
                prev_dog, curr_dog, next_dog = dog_images[index][i - 1], dog_images[index][i], dog_images[index][i + 1]


                for y in range(1, curr_dog.shape[0] - 1):
                    for x in range(1, curr_dog.shape[1] - 1):
                        compare_list = np.concatenate((
                            prev_dog[y-1 : y+2, x-1 : x+2].ravel(),
                            next_dog[y-1 : y+2, x-1 : x+2].ravel()
                        ), axis=None)

                        tmp_curr = curr_dog[y-1 : y+2, x-1 : x+2].ravel()
                        tmp_curr = np.delete(tmp_curr, 4)

                        compare_list = np.concatenate((compare_list, tmp_curr), axis=None)

                        if (curr_dog[y, x] <= np.min(compare_list) or curr_dog[y, x] >= np.max(compare_list)):
                            if (np.abs(curr_dog[y, x]) <= self.threshold):
                                continue
                            
                            keypoints.append([y << index, x << index])



        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)


        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
