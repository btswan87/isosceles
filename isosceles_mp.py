from __future__ import print_function
import gdal
import os
import numpy as np
from time import time
import osr
import argparse
import shutil
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import multiprocessing as mp
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
import pandas as pd
import string

y_block_size = 500
x_block_size = 500
n_bands = 4
null_pix_value = 0

# Prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
n_filters = len(kernels)


def bgr2grey(bgr):
    grey = (bgr[0, :, :] * 0.1140 + bgr[1, :, :] * 0.5870 + bgr[2, :, :] * 0.2989)
    return grey


def compute_filter_feats(arr, bank):
    feats = np.zeros((2 * n_filters), dtype=np.double)
    for k, item in enumerate(bank):
        filtered = ndi.convolve(arr, item, mode='wrap')
        feats[k] = filtered.mean()
        feats[k+n_filters] = filtered.var()
    return feats


def get_feats(io_arg):
    arr, chip_idx = io_arg
    arr_ds = gdal.Open(arr)
    chip_feats = np.zeros((1, 42))

    chip_feats[0][0] = chip_idx[1]
    chip_feats[0][1] = chip_idx[0]

    chip = arr_ds.ReadAsArray(chip_idx[1], chip_idx[0], 500, 500)

    for b in range(0, chip.shape[0]):
        chip_feats[0][b + 2] = chip[b, :, :].mean()
        chip_feats[0][b + 2 + chip.shape[0]] = chip[b, :, :].std()

    chip_feats[0][10:] = compute_filter_feats(bgr2grey(chip), kernels)

    return chip_feats


def get_chips(im_ras):
    samples = []
    arr = gdal.Open(im_ras)
    y_size = arr.RasterYSize
    x_size = arr.RasterXSize
    for y in range(0, y_size, y_block_size):
        if y + y_block_size < y_size:
            rows = y_block_size
        else:
            continue
        for x in range(0, x_size, x_block_size):
            if x + x_block_size < x_size:
                cols = x_block_size
                block = arr.ReadAsArray(x, y, cols, rows)
                if null_pix_value not in block:
                    samples.append((y, x))

    return samples


def mp_feats(arr, chips, processes):
    chips = zip([arr] * len(chips), [i for i in chips])
    num_proc = processes
    pool = mp.Pool(processes=num_proc)
    mpresult = []
    mpresult.extend(pool.map(get_feats, chips))
    pool.close()
    pool.join()

    return mpresult


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Ready')
    parser.add_argument('--wd', dest='wd', required=True,
                        help='Working directory to store temp raster')
    parser.add_argument('--list', dest='filelist', required=True,
                        help='Path to text file listing images to process')
    parser.add_argument('--dst', dest='dst_folder', required=True,
                        help='Top level folder for output')
    parser.add_argument('--fn', dest='out_fn', required=True,
                        help='Prefix for output statistics files')
    parser.add_argument('--n_proc', dest='n_proc', required=True, type=int,
                        help='Number of worker processes to use')
    parser.add_argument('--pref', dest='pref', required=False, type=float,
                        help='AP preference value to be used instead /'
                             'of default')

    args = parser.parse_args()
    wd = args.wd
    filelist = args.filelist
    dst_folder = args.dst_folder
    out_fn = args.out_fn
    n_proc = args.n_proc
    pref = args.pref

    if pref:
        print("Preference set to {}".format(pref))
    else:
        print("Default preference will be used")

    # Create output folder if it doesn't already exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    scene_list = [line.rstrip('\n\r', ) for line in open(filelist)]

    for i in range(0, len(scene_list)):
        scene = scene_list[i]
        start_scene = time()
        im_basename = os.path.splitext(os.path.basename(scene))[0]
        im_folder = "ex_" + im_basename
        out_path = os.path.join(dst_folder, im_folder)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        os.chdir(out_path)

        temp_dst = os.path.join(wd, os.path.basename(scene))
        shutil.copy(scene, temp_dst)
        src_ds = gdal.Open(temp_dst)

        # Get raster parameters from sat image
        x_cell_size = src_ds.GetGeoTransform()[1]
        y_cell_size = src_ds.GetGeoTransform()[5]
        skew = 0
        x_min = src_ds.GetGeoTransform()[0]
        y_max = src_ds.GetGeoTransform()[3]
        wkt = src_ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)

        chips = get_chips(scene)

        start_blocks = time()
        print("Beginning scene {} of {}".format((i + 1), (len(scene_list))))

        result = mp_feats(temp_dst, chips, n_proc)

        minutes = ((time() - start_blocks) / 60)
        print("Scene scan finished in {} minutes".format(minutes))

        # Create numpy array to hold samples
        result_array = np.zeros((len(result), 42))

        for n in range(0, len(result)):
            result_array[n][:] = result[n]

        df = pd.DataFrame(data=result_array, columns=['x_origin', 'y_origin', 'blue_mean', 'green_mean',
                                                      'red_mean', 'nir_mean', 'blue_std', 'green_std',
                                                      'red_std', 'nir_std', 'f1_mean', 'f2_mean', 'f3_mean',
                                                      'f4_mean', 'f5_mean', 'f6_mean', 'f7_mean', 'f8_mean',
                                                      'f9_mean', 'f10_mean', 'f11_mean', 'f12_mean', 'f13_mean',
                                                      'f14_mean', 'f5_mean', 'f16_mean', 'f1_std', 'f2_std',
                                                      'f3_std', 'f4_std', 'f5_std', 'f6_std', 'f7_std',
                                                      'f8_std', 'f9_std', 'f10_std', 'f11_std', 'f12_std',
                                                      'f13_std', 'f14_std', 'f15_std', 'f16_std'])

        df.to_csv("{}_ex_{}.csv".format(out_fn, im_basename), index=False,
                  header=True)

        samples = result_array[:, 2:]

        sample_idxs = result_array[:, :2]

        # Initialize array to hold scaled samples
        scaled_samples = np.zeros(samples.shape)

        # Scale each feature type separately
        scaled_samples[:, 0:4] = preprocessing.scale(samples[:, 0:4])
        scaled_samples[:, 4:8] = preprocessing.scale(samples[:, 4:8])
        scaled_samples[:, 8:24] = preprocessing.scale(samples[:, 8:24])
        scaled_samples[:, 24:] = preprocessing.scale(samples[:, 24:])

        if pref:
            ap = AffinityPropagation(max_iter=10000, convergence_iter=100,
                                     preference=pref,
                                     affinity="euclidean").fit(scaled_samples)
        else:
            ap = AffinityPropagation(max_iter=10000, convergence_iter=100,
                                     affinity="euclidean").fit(scaled_samples)

        cluster_center_indices = ap.cluster_centers_indices_

        chip_count = 1

        for tile in cluster_center_indices:
            subset = sample_idxs[tile]
            path = os.path.join(out_path, im_basename + "_" + str(chip_count).zfill(4) + '.tif')

            # Get all raster bands for subset
            band = src_ds.GetRasterBand(1)
            red = band.ReadAsArray(int(subset[0]), int(subset[1]), x_block_size, y_block_size)
            band = src_ds.GetRasterBand(2)
            green = band.ReadAsArray(int(subset[0]), int(subset[1]), win_xsize=500, win_ysize=500)
            band = src_ds.GetRasterBand(3)
            blue = band.ReadAsArray(int(subset[0]), int(subset[1]), win_xsize=500, win_ysize=500)
            band = src_ds.GetRasterBand(4)
            nir = band.ReadAsArray(int(subset[0]), int(subset[1]), win_xsize=500, win_ysize=500)

            # Create geotransform information for destination raster
            x_origin = x_min + (subset[0] * x_cell_size)
            y_origin = y_max + (subset[1] * y_cell_size)
            new_transform = (x_origin, x_cell_size, skew, y_origin, skew, y_cell_size)

            # Create destination dataset
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(path, x_block_size,
                                   y_block_size, 4,
                                   gdal.GDT_UInt16)

            # Write subsetted bands to destination dataset
            dst_ds.GetRasterBand(1).WriteArray(red)
            dst_ds.GetRasterBand(2).WriteArray(green)
            dst_ds.GetRasterBand(3).WriteArray(blue)
            dst_ds.GetRasterBand(4).WriteArray(nir)

            # Set geotransform and projection info
            dst_ds.SetGeoTransform(new_transform)
            dst_ds.SetProjection(srs.ExportToWkt())

            # Close output
            dst_ds = None
            chip_count += 1

        src_ds = None
        os.remove(temp_dst)

        minutes = ((time() - start_scene) / 60)
        print("Scene finished in {} minutes".format(minutes))

