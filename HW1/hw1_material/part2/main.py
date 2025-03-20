import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def calculate(sigma_s, sigma_r, img_rgb, img_gray, index):
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_result = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_result = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    error = np.sum(np.abs(bf_result.astype('int32') - jbf_result.astype('int32')))
    # print(f'Error score: {error}')

    jbf_result = cv2.cvtColor(jbf_result, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f'filter_{index}.png', jbf_result)

def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        data = [list(map(float, line.strip().split(','))) for line in lines[1:-1]]
        sigma_parts = lines[-1].strip().split(",")
        sigma_s = int(sigma_parts[1])
        sigma_r = float(sigma_parts[3])

    return np.array(data), sigma_s, sigma_r

def gray_img(w, img_rgb, i):
    gray = w[0] * img_rgb[:,:,0] + w[1] * img_rgb[:,:,1] + w[2] * img_rgb[:,:,2]
    cv2.imwrite(f"gray_1_{i}.png", gray)
    return gray



def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ### 
    weights, sigma_s, sigma_r = read_file(args.setting_path)   

    


    for i, w in enumerate(weights):
        calculate(sigma_s, sigma_r, img_rgb, gray_img(w, img_rgb, i), i)

    calculate(sigma_s, sigma_r, img_rgb, img_gray, -1)
    

    


if __name__ == '__main__':
    main()