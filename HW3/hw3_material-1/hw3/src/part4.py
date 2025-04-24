import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

        src_pts = np.array([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.array([kp2[m.trainIdx].pt for m in matches])
        # TODO: 2. apply RANSAC to choose best H
        

        rand_idx_range = min(len(src_pts), len(dst_pts))
        H_best = np.eye(3)
        max_inlier = 0
        sample_kp = 10
        threshold = 0.5
 

        for i in range(1000):
            #  get an estimated H
            rand_idx = random.sample(range(rand_idx_range), sample_kp)
            est_H = solve_homography(dst_pts[rand_idx], src_pts[rand_idx])
            
            # H @ kp1 vs kp2
            dst_pts_homo = np.hstack([dst_pts, np.ones((len(dst_pts), 1))]).T
            est_src_pts_homo = est_H @ dst_pts_homo
            est_src_pts = (est_src_pts_homo[:2] / est_src_pts_homo[2]).T

            # caculate # of inlier points
            inliers = np.sum(np.linalg.norm(est_src_pts - src_pts, axis=1) <= threshold)

            # update best H
            if inliers > max_inlier:
                max_inlier = inliers
                H_best = est_H.copy()


         
        # TODO: 3. chain the homographies

        last_best_H = last_best_H @ H_best

        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, ymin=0 , ymax=h_max ,xmin=0 ,xmax=w_max , direction='b')

    return dst 


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)