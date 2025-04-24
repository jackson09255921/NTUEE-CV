import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
        # AH = 0
        # u [top-left, top-right, bottom-left, bottom-right
    A = np.array([
        #   11      12     13    21       22      23          31                   32             33
        [u[0][0], u[0][1], 1,    0   ,    0,      0,  -u[0][0] * v[0][0], -u[0][1] * v[0][0], -v[0][0]],
        [0,         0,     0, u[0][0], u[0][1],   1,  -u[0][0] * v[0][1], -u[0][1] * v[0][1], -v[0][1]],

        [u[1][0], u[1][1], 1,    0   ,    0,      0,  -u[1][0] * v[1][0], -u[1][1] * v[1][0], -v[1][0]],
        [0,         0,     0, u[1][0], u[1][1],   1,  -u[1][0] * v[1][1], -u[1][1] * v[1][1], -v[1][1]],

        [u[2][0], u[2][1], 1,    0   ,    0,      0,  -u[2][0] * v[2][0], -u[2][1] * v[2][0], -v[2][0]],
        [0,         0,     0, u[2][0], u[2][1],   1,  -u[2][0] * v[2][1], -u[2][1] * v[2][1], -v[2][1]],

        [u[3][0], u[3][1], 1,    0   ,    0,      0,  -u[3][0] * v[3][0], -u[3][1] * v[3][0], -v[3][0]],
        [0,         0,     0, u[3][0], u[3][1],   1,  -u[3][0] * v[3][1], -u[3][1] * v[3][1], -v[3][1]]
    ], dtype=float)


    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    H = VT[-1, :]

    H = H / H[-1] if H[-1] != 0 else H
    H = H.reshape(3, 3)

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    
    xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xy_homo = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())], axis=1)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        mapped = (H_inv @ xy_homo.T).T
        mapped /= mapped[:, 2:]
        h_patch, w_patch = ymax - ymin, xmax - xmin
        u = mapped[:, 0].reshape(h_patch, w_patch)
        v = mapped[:, 1].reshape(h_patch, w_patch)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = ((u >= 0) & (u < w_src - 1) & (v >= 0) & (v < h_src - 1))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking
        # dst[yy[mask], xx[mask]] = src[v[mask].astype(int), u[mask].astype(int)]
        dst[yy[mask], xx[mask]] = bilinear_interpolate(src, u[mask], v[mask])
        

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        mapped = (H @ xy_homo.T).T
        mapped /= mapped[:, 2:]

        h_patch, w_patch = ymax - ymin, xmax - xmin
        u = mapped[:, 0].reshape(h_patch, w_patch)
        v = mapped[:, 1].reshape(h_patch, w_patch)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (u >= 0) & (u < w_dst) & (v >= 0) & (v < h_dst)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        x_valid = u[mask].astype(int)
        y_valid = v[mask].astype(int)

        x_src = xx[mask]
        y_src = yy[mask]

        # TODO: 6. assign to destination image using advanced array indicing

        dst[y_valid, x_valid] = src[y_src, x_src]

    return dst 


def bilinear_interpolate(image, x, y):
    x1, y1 = np.floor(x).astype('int'), np.floor(y).astype('int')
    x2, y2 = x1 + 1, y1 + 1

    wa = np.repeat((y2 - y) * (x2 - x), 3).reshape((-1, 3))
    wb = np.repeat((x2 - x) * (y - y1), 3).reshape((-1, 3))
    wd = np.repeat((x - x1) * (y2 - y), 3).reshape((-1, 3))
    wc = np.repeat((x - x1) * (y - y1), 3).reshape((-1, 3))

    result = wa * image[y1, x1] + wb * image[y2, x1] + wc * image[y2, x2] + wd * image[y1, x2]

    return result