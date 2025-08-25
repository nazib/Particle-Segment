import numpy as np
import torch
from PIL import Image

#### SAM from SAM-github####
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry,SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
### SAM from hhuggingface #####
from transformers import SamModel, SamProcessor
from transformers import pipeline
from matplotlib import pyplot as plt
import gc
from skimage.measure import label, regionprops

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    plt.show()
    del mask
    gc.collect()

def run_on_cpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    #processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    #generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)

    img_url = r"C:\BHP\PT_test_export\images_20240805_120633\Multiple\ECU_901_902_CL_inverted_82_[x=4507,y=4037,w=270,h=230].tif"
    raw_image = Image.open(img_url).convert("RGB")
    #outputs = generator(raw_image, points_per_batch=64)
    #masks = outputs["masks"]
    sam = sam_model_registry["vit_b"](checkpoint=r"C:\BHP\Correlation_Analysis_data\code\SAM-models\sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(np.asarray(raw_image))
    #masks = SamOnnxModel(sam,return_single_mask=False)

    jmasks = []
    for mask in masks:
        jmasks.append(mask['segmentation'])

    show_masks_on_image(raw_image, masks)

def grid_points(num_points):
    x = np.linspace(0, 279, num=num_points, dtype=int)
    y = np.linspace(0, 279, num=num_points, dtype=int)
    xx,yy = np.meshgrid(x,y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    return grid_points

def run_on_onnx():
    img_url = r"C:\BHP\PT_test_export\images_20240805_120633\Multiple\ECU_901_902_CL_inverted_82_[x=4507,y=4037,w=270,h=230].tif"
    raw_image = np.asarray(Image.open(img_url).convert("RGB"))
    sam = sam_model_registry['vit_b'](checkpoint=r"C:\BHP\Correlation_Analysis_data\code\SAM-models\sam_vit_b_01ec64.pth")
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    predictor.set_image(raw_image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    image_embedding.shape
    
    #### for 32 random points ####
    input_point =grid_points(num_points=4) #np.random.randint(280,size=(32,2))
    input_label = np.random.randint(1,size=(input_point.shape[0],1))
    input_label[:,:] = 1
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([[-1]])], axis=0).astype(np.float32)
    '''
    #### for a single point ####
    input_point = np.array([[50, 50]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    '''
    onnx_coord = predictor.transform.apply_coords(onnx_coord, raw_image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    
    ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label.T,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(raw_image.shape[:2], dtype=np.float32)
                }
    
    onnx_model_path = r"C:\BHP\Correlation_Analysis_data\code\sam-vit-base-onnox\sam-vit-b-onnox"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    show_masks_on_image(raw_image,masks[0])
    print("Done")
    
if __name__== "__main__":
    run_on_cpu()
    print("Done!!!")
    
