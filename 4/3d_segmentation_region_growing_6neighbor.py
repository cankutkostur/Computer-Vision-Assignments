import numpy as np
import cv2
import nibabel as nib

# threshold
T = 0.5


def dice(seg, truth):
    intersect = 0
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                if seg[i, j, k] == truth[i, j, k] == 1:
                    intersect += 1

    return 2 * intersect / (np.sum(seg) + np.sum(truth))


def similar(i, j):
    return abs(i - j) < T


def neighbors_6(point, shape):
    neighbors = []

    for i in range(point[0] - 1, point[0] + 2):
        for j in range(point[1] - 1, point[1] + 2):
            for k in range(point[2] - 1, point[2] + 2):
                if i == point[0] and j == point[1] and k == point[2]:
                    continue
                same_index_count = np.sum(point == np.asarray([i, j, k]))
                if same_index_count != 2:
                    continue
                if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                    neighbors.append([i, j, k])

    return neighbors


def region_growing(data, seeds):
    segmented = np.zeros_like(data)

    for seed in seeds:
        expansion = [seed]
        while len(expansion) > 0:
            point = expansion.pop()
            if segmented[point[0], point[1], point[2]] == 0:
                segmented[point[0], point[1], point[2]] += 1
                neighbors = neighbors_6(point, data.shape)
                for neighbor in neighbors:
                    if similar(data[seed[0], seed[1], seed[2]], data[neighbor[0], neighbor[1], neighbor[2]]):
                        expansion.append(neighbor)

    return segmented


def main():
    img = nib.load("V.nii")
    img_seg = nib.load("V_seg_05.nii")

    data_img = np.asarray(img.get_fdata())
    data_seg = np.asarray(img_seg.get_fdata())

    f = open("3D_seed.txt", "r")

    seeds = []
    line = f.readline()
    while line:
        line = line.split()
        seed = []
        for i in line:
            seed.append(int(i))
        seeds.append(seed)
        line = f.readline()

    f.close()

    segmented = region_growing(data_img, seeds)
    print("Dice score: " + str(dice(segmented, data_seg)))

    new_img = nib.Nifti1Image(segmented, img.affine, img.header)
    nib.save(new_img, "part3d.nii")


if __name__ == "__main__":
    main()
