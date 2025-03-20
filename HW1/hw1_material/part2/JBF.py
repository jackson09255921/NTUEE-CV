
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros_like(img)
        Gs_LUT = np.exp(- (np.arange(self.pad_w+1)**2)/(2*self.sigma_s**2))
        Gr_LUT = np.exp(- ((np.arange(256)/255)**2)/(2*self.sigma_r**2))

        upper = np.zeros_like(padded_img, dtype=np.float64)
        lower = np.zeros_like(padded_img, dtype=np.float64)

        for y in range(-self.pad_w, self.pad_w+1):
            for x in range(-self.pad_w, self.pad_w+1):
                Gs = Gs_LUT[abs(x)] * Gs_LUT[abs(y)]

                if guidance.ndim == 2:
                    Gr = Gr_LUT[np.abs(np.roll(padded_guidance, shift=(y, x), axis=(0,1)) - padded_guidance)]
                else:
                    Gr = 1
                    for ch in range(3):
                        diff = np.abs(np.roll(padded_guidance[:,:,ch], shift=(y,x), axis=(0,1)) - padded_guidance[:,:,ch])
                        Gr *= Gr_LUT[diff]

                sr = Gs * Gr
                shifted_img = np.roll(padded_img, shift=(y, x), axis=(0,1))

                for ch in range(3):
                    upper[:,:,ch] += sr * shifted_img[:,:,ch]
                    lower[:,:,ch] += sr

        
        output = upper[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w] / lower[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w]
        return np.clip(output, 0, 255).astype(np.uint8)
