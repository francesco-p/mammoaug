#!/usr/bin/env python
# coding: utf-8

import glob
import random
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from skimage.exposure import match_histograms
from tqdm import tqdm
from collections import defaultdict


class TittyTooSmallForImplantEcxception(Exception):
    "Raised when the titty is smaller"
    pass


def crop_to_rectangular(img):
    """
    img: np.uint16 original image
    returns cropped image
    """
    thresh = (img>0).astype(np.uint8)
    x,y,h,w = get_biggest_blob_roi(thresh)
    img_cropped = img[y:y+h, x:x+w]
    return img_cropped


def get_biggest_blob_roi(img, debug=False):
    """
    (for reference y -> height dimension and x -> width dimension)
    thresh: np.uint8 thresholded image
    returns: the biggest blob roi
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if debug:
        src2 = np.zeros((img.shape[0], img.shape[1], 3))
        src2[:, :, 0] = img.astype(np.uint8)
        src2[:, :, 1] = img.astype(np.uint8)
        src2[:, :, 2] = img.astype(np.uint8)

        # cv2.drawContours(src2, contours[57], -1, (255, 0, 0), 30)
        cv2.circle(src2, (x+w//2, y+h//2), 50, (0, 0, 255), 50)
        print(f'{x=},{y=},{h=},{w=},{src2.shape=}')

        cv2.rectangle(src2, (x, y), (x+w, y+h), (255, 0, 0), 90)
        plt.imshow(src2)
        plt.axis('off')
        plt.show()

    return x, y, h, w


def apply_windowing(dicom_img):
    """
    Function that applies a windowing operation to a DICOM image.
    The function applies a window width of 1500 HU and centers it on the value specified in the DICOM header.
    The function also inverts the image if the PhotometricInterpretation is "MONOCHROME1"
   
    Parameters:
    dicom_img (pydicom.dataset.FileDataset) : a DICOM image.
    
    Returns:
    numpy.ndarray : the windowed image, as a numpy array.
    """
    
    # Extract the pixel array from the DICOM image
    img = dicom_img.pixel_array

    # Extract the window width and window center from the DICOM header
    if type(dicom_img.WindowWidth) == pydicom.valuerep.DSfloat:
        wwidth = int(dicom_img.WindowWidth)
        c = int(dicom_img.WindowCenter)
    else:
        wwidth = int(dicom_img.WindowWidth[0])
        c = int(dicom_img.WindowCenter[0])

    # Apply windowing only if the default window width (4096) is not used
    if wwidth != 4096:
        # Set the window width and window center
        w = 1500
        
        # Find pixels that fall outside the window
        pt0 = np.where(img < (c-w//2))
        ptm = np.where(img > (c+w//2))
        
        # Find pixels that fall within the window
        ptall = np.where((img <= (c+w//2)) & (img >= (c-w//2)))
        
        # Scale the pixel values to the range [0, 4096]
        img = (4096//w)*(img-(c-w//2))
        img = img.astype(np.uint16)
        
        # Set pixels that fall outside the window to 0 or the maximum value
        img[pt0] = 0
        img[ptm] = img[ptall].max()

    # Invert the image if the PhotometricInterpretation is "MONOCHROME1"
    if dicom_img.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    return img


def isolate_tumor(ill, mask):
    """
    ill image np.uint16
    tumor mask np.uint8
    returns pixel indices, pixel values, height and width of roi box
    of the original annotated mask
    """
    indices = np.where(mask > 0)
    tumor = ill[indices[0], indices[1]]
    x, y, h, w = get_biggest_blob_roi(mask)
    return indices, tumor, h, w


def get_tumor_patch(ill, mask):
    """
    ill image np.uint16
    tumor mask np.uint8
    returns pixel indices, pixel values, height and width of roi box
    of a square patch
    """
    tumor_mask = mask * ill
    tmp = (tumor_mask > 0).astype(np.uint8)
    x, y, h, w = get_biggest_blob_roi(tmp)
    patch = ill[y:y+h, x:x+w]

    tmp = np.zeros(ill.shape)
    tmp[y:y+h, x:x+w] = 1
    indices = np.nonzero(tmp)
    return indices, patch.flatten(), h, w


def implant_tumor(sane, ill, ill_mask, patched=True, dspl=200, debug=False):

    # Resize ill and mask to sane shape
    resized_mask = (cv2.resize(ill_mask, sane.shape[::-1], interpolation=cv2.INTER_AREA) / 255).astype(np.uint8)
    resized_ill = cv2.resize(ill, sane.shape[::-1], interpolation=cv2.INTER_AREA)

    # Normalize in 0-1
    normalized_ill = resized_ill / resized_ill.max()
    normalized_sane = sane / sane.max()

    # Find sane titty center
    img = (sane > 0).astype(np.uint8)
    x, y, h, w = get_biggest_blob_roi(img, debug=False)
    cx, cy = (x+w//2, y+h//2)

    if w < img.shape[1]*0.3:
        raise TittyTooSmallForImplantEcxception

    # displace sane target center
    cx, cy = cx + np.random.randint(-dspl, dspl), cy+np.random.randint(-dspl, dspl)

    # Get tumor from ill
    if not patched:
        indices, ill_patch, patch_h, patch_w = isolate_tumor(normalized_ill, resized_mask)
    else:
        indices, ill_patch, patch_h, patch_w = get_tumor_patch(normalized_ill, resized_mask)

    # indices of tumor in ill cohordinates
    y_offsets, x_offsets = indices[0]-indices[0].min(), indices[1]-indices[1].min()

    # indices of tumor in sane cohordinates (it filters the indices if it's outside sane image shape)
    y_trg = cy+y_offsets
    y_filtered_target_indices = y_trg < normalized_sane.shape[0]

    x_trg = cx+x_offsets
    x_filtered_target_indices = x_trg < normalized_sane.shape[1]

    flter = y_filtered_target_indices & x_filtered_target_indices
    x_trg = x_trg[flter]
    y_trg = y_trg[flter]

    # Brvtal implant
    # normalized_sane[cy+y_offsets, cx+x_offsets] = tumor

    # Hist match and implant
    sane_patch = normalized_sane[y_trg, x_trg]
    matched_ill_patch = match_histograms(ill_patch, sane_patch, channel_axis=None)#, multichannel=False)

    # print(matched_ill_patch.shape, sane_patch.shape)

    # DBUG
    if not debug:

        normalized_sane[y_trg, x_trg] = matched_ill_patch[flter]

    else:

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))

        axs[0].set_title('sane')
        axs[0].imshow(normalized_sane)
        axs[0].set_axis_off()
        cv2.imwrite('sane.png', normalized_sane*255)

        normalized_sane[y_trg, x_trg] = ill_patch[flter]
        axs[1].set_title('implanted_nohist')
        axs[1].imshow(normalized_sane)
        axs[1].set_axis_off()
        cv2.imwrite('implanted_nohist.png', normalized_sane*255)

        normalized_sane[y_trg, x_trg] = matched_ill_patch[flter]
        axs[2].set_title('implanted_hist')
        axs[2].imshow(normalized_sane)
        axs[2].set_axis_off()
        cv2.imwrite('implanted_hist.png', normalized_sane*255)

        masked = resized_mask * 0.5
        masked += normalized_ill
        axs[3].set_title('ill')
        axs[3].imshow(masked)
        axs[3].set_axis_off()
        cv2.imwrite('ill.png', masked*255)

        # axs[4].set_title('sane_patch')
        # img = np.zeros(patch_h, patch_w)
        # axs[4].imshow(img[flter].reshape(patch_h, patch_w))
        # axs[4].set_axis_off()
        # cv2.imwrite('sane_patch.png', masked*255)

        # axs[5].set_title('ill_patch')
        # axs[5].imshow(matched_ill_patch[flter].reshape(patch_h, patch_w))
        # axs[5].set_axis_off()
        # cv2.imwrite('ill_patch.png', masked*255)

        plt.show()
    
    # Cropped to rectangle
    normalized_sane = normalized_sane[y:y+h, x:x+w]
    return normalized_sane


def extract_ill_sanes_from_datasets(challenge_dset_path, annotated_dset_path, mass_and_calc=False):

    # Dataset to be augmented
    challenge_images = glob.glob(f'{challenge_dset_path}/train_images/*/*.dcm')
    challenge_df = pd.read_csv(f'{challenge_dset_path}/train.csv')

    # Get all ill and sane image files of challenge dataset
    sane_patients_id = np.unique(challenge_df[(challenge_df['cancer'] == 0) & (challenge_df['implant'] == 0)]['patient_id'].values).tolist()

    sane_img_paths = [image_path for image_path in challenge_images if int(image_path.split('/')[-2]) in sane_patients_id]

    # Annotated dataset
    if mass_and_calc:
        calc_df = pd.read_csv(f'{annotated_dset_path}/calc_case_description_train_set.csv')
        mass_df = pd.read_csv(f'{annotated_dset_path}/mass_case_description_train_set.csv')
        annotated_df = pd.concat([calc_df, mass_df], ignore_index=True, axis=0)
    else:
        # Only calc
        annotated_df = pd.read_csv(f'{annotated_dset_path}/calc_case_description_train_set.csv')

    # MLO or CC?
    annotated_images = glob.glob(f'{annotated_dset_path}/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/*_CC/*/*/*.dcm')
    annotated_images += glob.glob(f'{annotated_dset_path}/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/*_MLO/*/*/*.dcm')
    annotated_masks = glob.glob(f'{annotated_dset_path}/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/*_1/*/*/*.dcm')

    ill_patients_id = np.unique(annotated_df[annotated_df['pathology'] == 'MALIGNANT']['patient_id'].values).tolist()
    ill_patients_id = [int(x.split('_')[1]) for x in ill_patients_id]

    ill_img_paths = [image_path for image_path in annotated_images if int(image_path.split('/')[-4].split('_')[-3]) in ill_patients_id]

    ill_mask_paths = [mask_path for mask_path in annotated_masks if int(mask_path.split('/')[-4].split('_')[-4]) in ill_patients_id]

    return (sane_patients_id, sane_img_paths), (ill_patients_id, ill_img_paths, ill_mask_paths)


def generate_malignant(ill_id, ill_img_paths, ill_mask_paths, sane_img_paths):
    OUT_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "malignant_augmented")

    print(ill_id)
    already_seen_ill_patients = defaultdict(int)

    # For each ill patient pick a random sane titty and implant tumor
    #for ill_id in tqdm(ill_patients_id):

    # Extract all paths of all ill images
    ill_path = [x for x in ill_img_paths if f'P_{ill_id:05d}' in x]
    ill = pydicom.dcmread(ill_path[0]).pixel_array

    # Make sure to select the mask file and not the patch file
    mask = None
    for path in [x for x in ill_mask_paths if f'P_{ill_id:05d}' in x]:
        mask = pydicom.dcmread(path).pixel_array
        if mask.shape == ill.shape:
            break
    
    if mask is None:
        print(f'None Mask {ill_id}, {ill_path}')
        return

    # Select random sane image
    sane_path = random.choice(sane_img_paths)
    sane_dicom_img = pydicom.dcmread(sane_path)
    sane = apply_windowing(sane_dicom_img)

    #print(f'{ill.shape=}, {mask.shape=}, {sane.shape=}')

    try:
        # Implant tumor and adjust dtype
        augmented = implant_tumor(sane, ill, mask, patched=True, dspl=100, debug=False)        
        augmented *= 4096
        augmented = augmented.astype(np.uint16)
        
        # Calculate new name I keep the dict because 
        # I want to keep track of how many sick images I have for each patient
        img_id = already_seen_ill_patients[ill_id]
        already_seen_ill_patients[ill_id] += 1
        name = f'{MAL_OUT_PATH}/{ill_id}_{img_id}.png'
        
        cv2.imwrite(name, augmented)

    except TittyTooSmallForImplantEcxception:
        print('Titty is too small for implant')

# to be optimal the patch size should be around the mean tumor size...
def generate_benign(start_id, step, patch_size = (200, 200), dspl=200):
    CHALLENGE_DSET_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "challenge_dset")
    OUT_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "benign_augmented")
    # All images name start from index 3000 just to discriminate from malignant
    # example output filename: 00_3001.jpg ({FAKE_PATIENT_ID}_{img_index+OFFSET}.jpg)
    FAKE_PATIENT_ID = 0
    OFFSET = 3000
    
    for img_id in range(start_id, start_id+step):
        print(f'{start_id} - {start_id+step} -> {img_id}')

        # select two random benign images and preprocess them through windowing
        imgs= glob.glob(f'{CHALLENGE_DSET_PATH}/train_images/*/*.dcm')
        img_1 = apply_windowing(pydicom.dcmread(imgs.pop(random.randrange(len(imgs)))))
        img_2 = apply_windowing(pydicom.dcmread(imgs.pop(random.randrange(len(imgs)))))

        # crop a random patch from img_1 and paste randomly to img_2
        bimg_1 = (img_1 > 0).astype(np.uint8)
        x, y, h, w = get_biggest_blob_roi(bimg_1)
        cx, cy = (x+w//2, y+h//2)

        if w < img_1.shape[1]*0.3:
            #raise TittyTooSmallForImplantEcxception
            continue

        # displace sane target center
        cx, cy = cx+np.random.randint(-dspl, dspl), cy+np.random.randint(-dspl, dspl)

        src_patch = img_1[cy:cy+patch_size[1], cx:cx+patch_size[0]]

        # crop a random patch from img_1 and paste randomly to img_2
        bimg_2 = (img_2 > 0).astype(np.uint8)
        x, y, h, w = get_biggest_blob_roi(bimg_2)
        cx, cy = (x+w//2, y+h//2)

        if w < img_2.shape[1]*0.3:
            #raise TittyTooSmallForImplantEcxception
            continue

        # displace sane target center
        cx, cy = cx+np.random.randint(-dspl, dspl), cy+np.random.randint(-dspl, dspl)

        trg_patch = img_2[cy:cy+patch_size[1], cx:cx+patch_size[0]]

        # Before computing histogram check shape
        if src_patch.shape != trg_patch.shape:
            print(f'Size Mismatch {src_patch.shape} != {trg_patch.shape}')
            continue            
        
        # Match histograms and replace patch
        matched_src_patch = match_histograms(src_patch, trg_patch, channel_axis=None)#, multichannel=False)
        img_2[cy:cy+patch_size[1], cx:cx+patch_size[0]] = matched_src_patch

        # Save images
        img_2 = crop_to_rectangular(img_2)
        cv2.imwrite(f'{BEN_OUT_PATH}/{FAKE_PATIENT_ID}_{img_id+OFFSET}.png', img_2)


######################################################
######################################################
######################################################


if __name__ == '__main__':

    MAL_OUT_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "malignant_augmented")
    BEN_OUT_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "benign_augmented")

    # check if folder is present and create it if not
    if not os.path.exists(MAL_OUT_PATH):
        os.makedirs(MAL_OUT_PATH)
    # check if folder is present and create it if not
    if not os.path.exists(BEN_OUT_PATH):
        os.makedirs(BEN_OUT_PATH)
       
    
    # Generates malignant images [in parallel] (IN TOTAL 590)
    MASS_AND_CALC = True
    CHALLENGE_DATASET_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "challenge_dset")
    ANNOTATED_DATASET_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "second_dset")
    


    (sane_patients_id, sane_img_paths), (ill_patients_id, ill_img_paths, ill_mask_paths) = extract_ill_sanes_from_datasets(CHALLENGE_DATASET_PATH, ANNOTATED_DATASET_PATH, MASS_AND_CALC)
    n_ill = len(ill_patients_id)
    _ = Parallel(n_jobs=-1)(delayed(generate_malignant)(ill_id, ill_img_paths, ill_mask_paths, sane_img_paths) \
        for (ill_id, ill_img_paths, ill_mask_paths, sane_img_paths) in zip(ill_patients_id, [ill_img_paths]*n_ill, [ill_mask_paths]*n_ill, [sane_img_paths]*n_ill))

    print(" GENERATION OF MALIGNANT IMAGES FINISHED, START BENIGN GENERATION ")
    
    ## Generates benign images [in parallel](IN TOTAL 600)
    #n_processes = 15
    #examples_per_process = 35
    #l = range(0, n_processes*examples_per_process, examples_per_process)
    #_ = Parallel(n_jobs=n_processes)(delayed(generate_benign)(start_id, step)for (start_id,step) in zip([x for x in l],[examples_per_process]*n_processes))

    
