import pydicom
import numpy as np
import pandas as pd
import cv2
import glob
from joblib import Parallel, delayed
import os

def get_biggest_blob_roi(img, debug=False):
    """
    thresh: np.uint8 thresholded image
    returns: the biggest blob roi
    """
    contours, hiear = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)

    return x,y,h,w


def crop_to_rectangular(img):
    """
    img: np.uint16 original image
    returns cropped image
    """
    thresh = (img>0).astype(np.uint8)
    x,y,h,w = get_biggest_blob_roi(thresh)
    img_cropped = img[y:y+h, x:x+w]
    return img_cropped


def apply_resize(img):
    height, width = img.shape
    if height > width:
        img = cv2.resize(img, (512, 1024))
    else:
        img = cv2.resize(img, (1024, 512))
    return img

def apply_windowing(sane_dicom_img):

    sane = sane_dicom_img.pixel_array
    sane = sane.astype(np.uint16)
    sane = sane[30:-30, 30:-30]
    
    if type(sane_dicom_img.WindowWidth) ==  pydicom.valuerep.DSfloat:
        wwidth = int(sane_dicom_img.WindowWidth)
        c = int(sane_dicom_img.WindowCenter)    

    else:
        wwidth = int(sane_dicom_img.WindowWidth[0])
        c = int(sane_dicom_img.WindowCenter[0])    

    if wwidth != 4096:
        
        w = 1500    
        
        pt0 = np.where(sane<(c-w//2))
        ptm = np.where(sane>(c+w//2))
        ptall = np.where((sane<=(c+w//2)) & (sane>=(c-w//2)))
        sane = (4096//w)*(sane-(c-w//2))
        sane = sane.astype(np.uint16)
        
        sane[pt0] = 0
        sane[ptm] = sane[ptall].max() 

    if sane_dicom_img.PhotometricInterpretation == "MONOCHROME1":
        sane = sane.max() - sane

    return sane


def main(dcm_path):
    dicom = pydicom.dcmread(dcm_path)
    img = apply_windowing(dicom)
    #print('window',img.shape, img.dtype, img.min(), img.max())
    #img = crop_to_rectangular(img)
    #print('rect',img.shape, img.dtype, img.min(), img.max())
    
    PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "TEST")

    image_id = dcm_path.split('/')[-1][:-4]
    patient_id = dcm_path.split('/')[-2]
    #img = apply_resize(img)
    #print('resz',img.shape, img.dtype, img.min(), img.max())
    cv2.imwrite(PATH+f'/{patient_id}_{image_id}.png', img)

if __name__ == '__main__':

    CHALLENGE_DATASET_PATH = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "challenge_dset")

    challenge_images= sorted(glob.glob(f'{CHALLENGE_DATASET_PATH}/train_images/*/*.dcm'))

    # for i, dcm_path in enumerate(challenge_images[306:]):
    #     print(i)
    #     main(dcm_path)

    _ = Parallel(n_jobs=-1, verbose=1)(delayed(main)(patient_id)for patient_id in challenge_images)