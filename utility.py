import numpy as np
from PIL import Image
import cv2
from skimage import io, filters, morphology, measure
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.
    
    Args:
    - mask1 (np.array): First binary mask.
    - mask2 (np.array): Second binary mask.
    
    Returns:
    - iou (float): The IoU score.
    """
    # Ensure that the masks are boolean (True/False) or binary (0/1)
    mask1 = mask1.astype(np.bool_)
    mask2 = mask2.astype(np.bool_)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    
    return iou


def show_masks_on_image(raw_image, masks):
    combind_image = np.asarray(raw_image).copy()
    all_mask_images =[]
    multi_mask = np.zeros(combind_image.shape,dtype=np.uint8)
    for i,mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.asarray(raw_image).copy()
        color = np.random.randint(0,255,3).tolist()
        contour_image = cv2.drawContours(contour_image, contours, -1,(0,255,0), 2)
        combind_image = cv2.drawContours(combind_image, contours, -1,color, 2)
        multi_mask[mask>0] = 255  #//(i+1)
        all_mask_images.append(contour_image)
    
    #all_mask_images.append(combind_image)
    return all_mask_images, multi_mask

def masks_filter(raw_image,sam_outputs):
    masks = []
    ##### Masks filtering based on area and boundingbox from SAM ####
    for seg in sam_outputs:
        w = seg['bbox'][2]
        h = seg['bbox'][3]
        if (raw_image.size[0] - w)<=5 and (raw_image.size[1] - h)<=5:
            continue
        elif seg['area'] <300:
            continue
        else:
            masks.append(seg['segmentation'])

    ##### Background Mask filtering using countour ####
    image = np.asarray(raw_image.convert("L")).copy()
    image = image.astype(float)
    image =  ndimage.median_filter(image, size=3)
    filtered_msk =[]
    kernel = np.ones((3, 3))
    for msk in masks:
        msk = msk.astype(np.uint8)
        msk_er =  ndimage.binary_erosion(msk, structure=kernel)
        contour = msk - msk_er
        filtered = image * msk
        filtered[contour==1]= 0
        if filtered.sum() >0:
            filtered_msk.append(msk)
    return filtered_msk

def detect_masks(raw_image,sigma=3, threshold=30):
    """
    Detect masks from SAM outputs and filter them based on area and bounding box.
    
    Args:
    - raw_image (PIL.Image): The original image.
   
    
    Returns:
    - filtered_masks (list): List of filtered binary masks.
    """
    # Convert raw_image to RGB if it's not already
    if not isinstance(raw_image, Image.Image):
        raw_image = Image.fromarray(raw_image)
        
    if raw_image.mode != 'L':
        raw_image = np.asarray(raw_image.convert('L'))
    
    img_smooth = filters.gaussian(raw_image, sigma)
    if threshold is None or threshold == 0:
        threshold = filters.threshold_otsu(img_smooth)
    else:
        threshold = threshold / 255.0
    
    binary = img_smooth > threshold   # foreground = True
    # 4. Remove tiny noise and fill holes
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = ndi.binary_fill_holes(binary)
    # 5. Label objects
    labeled, num = ndi.label(binary)
    #binary_image = raw_image.point(lambda p: 255 if p > 30 else 0)
    #binary_image = np.array(binary_image, dtype=np.uint8)
    
    regions = regionprops(labeled)
    regions = [region for region in regions if region.area > 300]
    masks = []
    for region in regions:
        msk = np.zeros(binary.shape, dtype=np.uint8)
        msk[region.coords[:, 0], region.coords[:, 1]] = 1
        masks.append(msk)
    return masks

def extract_single_mask(raw_image):
    """
    Extract a single mask from the raw image.
    
    Args:
    - raw_image (PIL.Image): The original image.
    - mask (np.array): The binary mask to extract.
    
    Returns:
    - extracted_image (PIL.Image): The image with the mask applied.
    """
    if raw_image.mode != 'L':
        binary = np.asarray(raw_image.convert('L').point(lambda p: 255 if p > 30 else 0))
    
    # 4. Remove tiny noise and fill holes
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = ndi.binary_fill_holes(binary)

    # 5. Label objects
    labeled, num = ndi.label(binary)
    mask =[]
    regions = regionprops(labeled)
    region = regions[0]  # Assuming we want the first region
    msk = np.zeros(binary.shape, dtype=np.uint8)
    msk[region.coords[:, 0], region.coords[:, 1]] = 255
    mask.append(msk)
    return mask
