#!/usr/bin/env python3

# CRITICAL: Import Qt FIRST before any AI libraries
import sys
import os

# Set Qt environment variables BEFORE any imports
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # Let Qt auto-detect
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering

# Import PyQt6 FIRST - this is critical!
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QGridLayout
from PyQt6.QtCore import QTimer

print("✓ PyQt6 imported successfully")

# Basic Python libraries (safe)
import numpy as np
import time
import glob
import json
import pandas as pd
import cv2
from PIL import Image, ImageQt
from PIL.ImageQt import toqimage, toqpixmap
from skimage.measure import label, regionprops
from torchvision import transforms
from keras.models import load_model
from utility import *
print("✓ Basic libraries imported")

from classification import *
# Load the UI file
ui_file = "Particle-Segment.ui"

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        print("Initializing MyApp...")
        
        # Load UI first
        try:
            uic.loadUi(ui_file, self)
            print("✓ UI loaded successfully")
        except Exception as e:
            print(f"✗ UI loading failed: {e}")
            sys.exit(1)
        
        # Initialize UI components
        self.initUI()
        
        # Initialize variables
        self.classification_model = None
        self.masks = []
        self.mask_images = []
        self.selected_mask = None
        self.counter = 0
        self.raw_image = None
        self.masked_image = None
        self.input_name_list = []
        self.output_folder_path = ""
        self.classification_model_path = "classification_models"
        self.norm_param_file = "classification_models/normalization_params.json"
        self.img_mn,self.img_std = load_normalization_params(self.norm_param_file)
        self.Normalize = transforms.Normalize(mean=self.img_mn, std=self.img_std)
        self.isClassified = False
        self.isMaskApplied = False
        self.fastSAM = None
        self.fastSAM_loaded = False
        self.curr_class_id = 0
        self.curr_probability = 0.0
        print("✓ Basic initialization complete")
        
        # Load AI libraries AFTER Qt is fully initialized
        self.ai_libraries_loaded = False
        self.load_ai_libraries()
        
        # CSV setup
        self.setup_csv_data()
        
        print("✓ MyApp initialization complete")
    
    def reset_variales(self):
        """Reset all variables to their initial state"""
        self.masks = []
        self.mask_images = []
        self.selected_mask = None
        self.raw_image = None
        self.masked_image = None
        self.isClassified = False
        self.class_name= ""
        self.isMaskApplied = False
        self.fastSAM = None
        self.fastSAM_loaded = False
        self.curr_class_id = 0
        self.curr_probability = 0.0

    def load_ai_libraries(self):
        """Load AI libraries after Qt is initialized"""
        try:
            print("Loading AI libraries...")
            
            # Import PyTorch with care
            global torch
            import torch
            print(f"✓ PyTorch loaded - CUDA: {torch.cuda.is_available()}")
            
            # Import segment_anything
            #global SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
            #from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
            #print("✓ segment_anything loaded")
            
            # Import ultralytics
            global FastSAM
            from ultralytics import FastSAM
            print("✓ ultralytics loaded")
            
            # Import utility
            global show_masks_on_image, masks_filter
            from utility import show_masks_on_image, masks_filter
            print("✓ utility loaded")
            
            self.ai_libraries_loaded = True
            print("🎉 All AI libraries loaded successfully!")
            
        except Exception as e:
            print(f"✗ AI library loading failed: {e}")
            QMessageBox.warning(self, "Warning", 
                              f"AI libraries failed to load: {str(e)}\n"
                              "Some features may not work.")
            import traceback
            traceback.print_exc()

    def setup_csv_data(self):
        """Setup CSV data handling"""
        try:
            columns = ['FileIndex','FileName','Time','ClassID', 
                       'Probability','ClassName',
                       'MasksApplied']
            
            self.csv_data = pd.DataFrame(columns=columns)
            self.csv_file_name = "SegmentationStatus.csv"
            
            if os.path.exists(self.csv_file_name):
                self.csv_data = pd.read_csv(self.csv_file_name)
                self.image_counter = self.csv_data['FileIndex'].iloc[-1]+1
                self.dict = self.csv_data.iloc[-1].to_dict()

                with open('metadata.json', 'r') as meta_file:
                    self.csv_data.attrs = json.load(meta_file)

                self.input_dir = self.csv_data.attrs["InputFolderPath"]
                self.input_dir_box.setText(self.input_dir)
                self.output_folder_path = self.csv_data.attrs['OutputFolderPath']
                self.out_dir_box.setText(self.output_folder_path)
                self.input_name_list = self.csv_data.attrs['InputNameList']
                self.display_folderInfo()
                QTimer.singleShot(0, self.load_input_image)
            else:
                self.csv_data.to_csv(self.csv_file_name, index=False)
                self.input_dir = ""
                self.dict = {}
                self.image_counter = 0
                
        except Exception as e:
            print(f"CSV setup error: {e}")
            self.input_dir = ""
            self.dict = {}
            self.image_counter = 0

    def initUI(self):
        try:
            self.detected_contours = 0
            
            # Connect basic signals
            self.SelectMain_btn.clicked.connect(self.select_input_folder_dialog)
            self.ExitMain_btn.clicked.connect(self.close_application)
            self.SelectOutput_btn.clicked.connect(self.select_output_folder_dialog)
            self.Classify_btn.clicked.connect(self.classify)
            self.SaveImage_btn.clicked.connect(self.SaveImage)
            self.LoadNext_btn.clicked.connect(self.LoadNext)
            self.Mask_btn.clicked.connect(self.masks_refine)
            self.MaksApply_btn.clicked.connect(self.apply_mask_on_image)
            self.LoadPrevious_btn.clicked.connect(self.LoadPrevious)
            # connects sliders
            self.Sigma_label.setText(f"Gaussian Sigma: {0}")
            self.Thrashold_label.setText(f"Binary Threshold: {0}")
            self.SigmaSlider.valueChanged.connect(self.on_sigma_changed)
            self.BinaryThrashSlider.valueChanged.connect(self.on_threshold_changed)
            self.sigma_value = self.SigmaSlider.value()
            self.threshold_value = self.BinaryThrashSlider.value()
            #self.SAMBox.stateChanged.connect(self.masks_refine)

            # Initialize combo boxes
            self.Prob_label.setText(f"Probability: {0.0}")
            self.ClassId_label.setText(f"Class ID: {0}")
            self.maskSelectBox.addItems([f"No Masks Detected"])
            self.modelSelectBox.addItems(["Resnet50","ConvNext","MobileNet","EfficientNet","DenseNet"])
            self.modelSelectBox.currentIndexChanged.connect(self.load_model_if_needed)
            #self.SamParamBox.addItems(['4','8','16','32'])
            self.ClassSelectBox.addItems(list(MaskProcess.color2class.keys()))
            
            # Connect combo box events
            self.maskSelectBox.currentIndexChanged.connect(self.scroll_image)
            self.ClassSelectBox.currentIndexChanged.connect(self.see_class_probabilities)
            
            print("✓ UI initialization complete")
            
        except Exception as e:
            print(f"✗ UI initialization failed: {e}")
            QMessageBox.critical(self, "UI Error", f"Failed to initialize UI: {str(e)}")

    def load_model_if_needed(self,value):
        """Lazy load models only when needed"""
        model_type = self.modelSelectBox.currentText()
        try:
            if not self.ai_libraries_loaded:
                QMessageBox.warning(self, "Error", "AI libraries not loaded. Cannot load models.")
                return False
            self.classification_model = MultiNetworkModel(
                        model_type,
                        num_classes=len(MaskProcess.color2class),
                        dropout_prob=0.2,
                        internal_dropout_prob=0.2,
                        full_train=False
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classification_model.to(device)

            # Load checkpoint if exists
            checkpoint_pattern = os.path.join(self.classification_model_path,model_type, f"best_model_*.pth")
            checkpoints = glob.glob(checkpoint_pattern)
            if checkpoints:
                try:
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    self.classification_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
                    print(f"Loaded {model_type}: {latest_checkpoint}")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
           
            return True
            
        except Exception as e:
            print(f"Model loading error: {e}")
            QMessageBox.critical(self, "Model Error", f"Failed to load {model_type}: {str(e)}")
            return False

    def close_application(self):
        self.close()
    
    def on_sigma_changed(self, value):
        """Handle sigma slider changes"""
        self.sigma_value = value
        self.Sigma_label.setText(f"Gaussian Sigma: {self.sigma_value}")
        
    def on_threshold_changed(self, value):
        """Handle threshold slider changes"""
        self.threshold_value = value
        self.Thrashold_label.setText(f"Binary Threshold: {self.threshold_value}")
        

    def select_input_folder_dialog(self):
        try:
            file_ext = ["*.tif","*.jpg","*.png","*.jpeg","*.JPEG","*.PNG"]
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

            if folder_path:
                self.input_dir = folder_path
                self.input_dir_box.setText(folder_path)
                
                self.input_name_list = []
                for ext in file_ext:
                    found_files = glob.glob(os.path.join(self.input_dir, ext))
                    if found_files:
                        self.input_name_list.extend(found_files)
                        
                if not self.input_name_list:
                    QMessageBox.warning(self,"No images found","Supported formats: tif, jpg, png, jpeg")
                    return
                    
                self.input_name_list.sort()
                self.image_counter = 0
                self.load_input_image()
                
                if not hasattr(self.csv_data, 'attrs'):
                    self.csv_data.attrs = {}
                self.csv_data.attrs["InputFolderPath"] = folder_path
                self.csv_data.attrs["InputNameList"] = self.input_name_list
                
        except Exception as e:
            print(f"Folder selection error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to select folder: {str(e)}")

    def display_folderInfo(self):
        try:
            if hasattr(self, 'input_name_list') and self.input_name_list:
                self.info = (
                    f"Input Dir: {os.path.basename(self.input_dir)}\n"
                    f"Output Dir: {os.path.basename(self.output_folder_path) if self.output_folder_path else 'Not set'}\n"
                    f"Total: {len(self.input_name_list)} Images\n"
                    f"Current: {os.path.basename(self.input_name_list[self.image_counter])}\n"
                    f"Processed: {self.image_counter}\n"
                    f"Classified: {self.isClassified}\n"
                    f"Mask Applied: {self.isMaskApplied}\n"
                    f"Current Class ID: {self.curr_class_id if self.isClassified else 'N/A'}\n"
                    f"Class Name: {self.class_name}\n"
                    f"Current Probability: {self.curr_probability:.2f}\n"
                    f"Mask Found: {len(self.masks)}\n"
                )
                self.FolderInfo.setText(self.info)
                
                if not hasattr(self, 'dict'):
                    self.dict = {}
                self.dict['FileName'] = os.path.basename(self.input_name_list[self.image_counter])
                self.dict['FileIndex'] = self.image_counter
        except Exception as e:
            print(f"Display folder info error: {e}")

    def load_input_image(self):
        try:
            if not hasattr(self, 'input_name_list') or not self.input_name_list:
                return
                
            if self.image_counter >= len(self.input_name_list):
                QMessageBox.information(self, "Info", "No more images")
                return
                
            image_path = self.input_name_list[self.image_counter]
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Warning", f"Image not found: {image_path}")
                return
                
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "Warning", f"Failed to load: {image_path}")
                return
            self.raw_image = Image.open(self.input_name_list[self.image_counter]).convert("RGB")    
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.inputView.setScene(scene)
            self.inputView.fitInView(scene.itemsBoundingRect())
            
            self.display_folderInfo()
            
        except Exception as e:
            print(f"Load image error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def classify(self):
        self.curr_probability = 0.0
        self.curr_class_id = 0
        self.isClassified = False
        index = 0
        try:
            if not self.ai_libraries_loaded:
                QMessageBox.critical(self, "Error", "AI libraries not loaded")
                return
                
            if not hasattr(self, 'input_name_list') or not self.input_name_list:
                QMessageBox.warning(self, "Warning", "Please select input folder first")
                return
                
            model_type = self.modelSelectBox.currentText()

            if self.classification_model is None:
                self.load_model_if_needed(model_type)
                
                
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if hasattr(self, 'raw_image') and self.raw_image is None:
                self.raw_image = Image.open(self.input_name_list[self.image_counter]).convert("RGB")
            
            if self.isMaskApplied and hasattr(self, 'masked_image') and self.masked_image is not None:
                image_array = self.masked_image.copy()
            else:
                image_array = np.asarray(self.raw_image)
                
            try:
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                image = Image.fromarray(image_array).resize((224, 224), Image.Resampling.LANCZOS)
                binary_model = load_model("./binary_model/keras_model.h5", compile=False)
                image = np.asarray(image)
                # Normalize the image
                normalized_image = (image.astype(np.float32) / 127.5) - 1
                data[0] = normalized_image
                # Predicts the model
                prediction = binary_model.predict(data)
                index = np.argmax(prediction)
            except Exception as e:
                print(f"Binary model prediction error: {e}")
                QMessageBox.critical(self, "Error", "Failed to predict using binary model")
                return
            
            if index == 1 and prediction[0][index] > 0.8:
                    reply = QMessageBox.question(
                        self, 
                        "Multiple Particles Detected", 
                        "Multiple Particles Detected /n Please Check SAM and try to to separate them.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No  # Default button
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        return

            if self.classification_model:
                st = time.time()
                input_image = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
                input_image = self.Normalize(input_image)
                outputs = self.classification_model(input_image)
                self.class_probabilities = torch.sigmoid(outputs).cpu().detach().numpy()[0]
                self.curr_probability =float(torch.sigmoid(outputs).cpu().detach().numpy()[0].max())
                self.curr_class_id = torch.argmax(outputs, dim=1).cpu().detach().numpy()[0]
                self.class_name = [ name for (name, code) in MaskProcess.color2class.items()][self.curr_class_id]
                self.Prob_label.setText(f"Prob: {self.curr_probability:.2f}")
                self.ClassId_label.setText(f"Class ID: {self.curr_class_id}")
                self.isClassified = True
                self.ClassSelectBox.setCurrentIndex(self.curr_class_id)
                en = time.time()
                ex_time = en - st
                #self.masks = detect_masks(image_array,self.sigma_value, self.threshold_value)
            else:
                QMessageBox.critical(self, "Error", "Classification model not loaded")
                return

            if self.masks:
                self.mask_images, self.multi_mask = show_masks_on_image(image_array, self.masks)
            
            # Update UI
            options = [f"Mask-{i+1}" for i in range(len(self.mask_images))]
            self.maskSelectBox.clear()
            self.maskSelectBox.addItems(options)
            
            self.display_folderInfo()
            
            if hasattr(self, 'ImageName'):
                self.ImageName.setText(os.path.basename(self.input_name_list[self.image_counter]))
            
            print("✓ Segmentation completed successfully")
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Segmentation Error", f"Segmentation failed: {str(e)}")

    
    def scroll_image(self, value):
        try:
            if not hasattr(self, 'masks') or not self.masks or not hasattr(self, 'mask_images'):
                return
            
            if 0 <= value < len(self.masks):
                self.selected_mask = self.masks[value]
                self.counter = value
                image_array = self.mask_images[value]
                image_array = Image.fromarray(image_array, "RGB")
                qimage = ImageQt.toqimage(image_array)
                pixmap = QPixmap.fromImage(qimage)
                scene = QGraphicsScene()
                scene.addPixmap(pixmap)
                self.intermView.setScene(scene)
                self.intermView.fitInView(scene.itemsBoundingRect())
                
        except Exception as e:
            print(f"Scroll image error: {e}")
    
    def apply_mask_on_image(self):
        try:
            if self.selected_mask is None:
                QMessageBox.warning(self, "Warning", "No mask selected")
                return
            
            if not hasattr(self, 'raw_image') or self.raw_image is None:
                QMessageBox.warning(self, "Warning", "No image loaded")
                return
            
            # Apply the selected mask to the raw image
            image = np.asarray(self.raw_image).copy()
            self.masked_image = np.zeros_like(np.asarray(image))
            self.masked_image[self.selected_mask>0] = image[self.selected_mask>0]
            
            # Convert to QPixmap and display
            qimage = ImageQt.toqimage(Image.fromarray(self.masked_image))
            pixmap = QPixmap.fromImage(qimage)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.intermView.setScene(scene)
            self.intermView.fitInView(scene.itemsBoundingRect())
            self.isMaskApplied = True
            
        except Exception as e:
            print(f"Apply mask error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply mask: {str(e)}")
        

    def select_output_folder_dialog(self):
        try:
            self.output_folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if self.output_folder_path:
                self.out_dir_box.setText(self.output_folder_path)
                self.display_folderInfo()
                self.image_out_dir = os.path.join(self.output_folder_path, "Images")
                self.label_out_dir = os.path.join(self.output_folder_path, "Labels")
                os.makedirs(self.image_out_dir, exist_ok=True)
                os.makedirs(self.label_out_dir, exist_ok=True)
                
                if not hasattr(self.csv_data, 'attrs'):
                    self.csv_data.attrs = {}
                self.csv_data.attrs['OutputFolderPath'] = self.output_folder_path
                
        except Exception as e:
            print(f"Output folder selection error: {e}")

    def SaveImage(self):
        try:
            if not hasattr(self, 'selected_mask') or self.selected_mask is None:
                QMessageBox.warning(self, "Warning", "No mask selected")
                return
                
            if not self.output_folder_path:
                QMessageBox.warning(self, "Warning", "Please select output folder first")
                return
            
            if self.isMaskApplied and self.masked_image is not None:
                cut_out = self.masked_image.copy()

            else:
                src_image = np.asarray(self.raw_image).copy()
                cut_out = np.zeros(shape=src_image.shape, dtype=src_image.dtype)
                cut_out[:,:] = src_image.min()
                cut_out[self.selected_mask>0] = src_image[self.selected_mask>0]
        
            file_name_parts = os.path.basename(self.input_name_list[self.image_counter]).split(".")
            file_name = f"{file_name_parts[0]}_{self.counter+1}.{file_name_parts[1]}"
            
            img_save_path = os.path.join(self.image_out_dir, file_name)
            lbl_save_path = os.path.join(self.label_out_dir, file_name)

            self.selected_mask[self.selected_mask > 0] = self.curr_class_id
            rgb_mask = MaskProcess.class2rgb(self.selected_mask)
            Image.fromarray(cut_out.astype(np.uint8)).save(img_save_path)
            
            Image.fromarray((rgb_mask).astype(np.uint8)).save(lbl_save_path)
            
            #QMessageBox.information(self, "Success", f"Image saved: {file_name}")
           
            if self.counter ==0 or len(self.csv_data)==0:
                self.dict["FileName"] = file_name
                self.dict["ClassID"] = str(self.curr_class_id)
                self.dict["Probability"] = str(self.curr_probability)
                self.dict["ClassName"] = self.class_name
                self.dict["FileIndex"] = self.image_counter
                self.dict["MasksApplied"] = str(self.isMaskApplied)
                new_data = pd.DataFrame(self.dict,index=[self.image_counter])
                self.csv_data = pd.concat([self.csv_data,new_data],ignore_index=True)
            else:
                df_slice = self.csv_data.iloc[self.image_counter].copy()
                df_slice["FileName"] = file_name
                df_slice["ClassID"] = str(self.curr_class_id)
                df_slice["Probability"] = str(self.curr_probability)
                df_slice["ClassName"] = self.class_name
                df_slice["FileIndex"] = self.image_counter
                df_slice["MasksApplied"] = str(self.isMaskApplied)
                self.csv_data.iloc[self.image_counter]= df_slice
                #self.csv_data['SavedLabels'].loc[self.image_counter]+= str(np.unique(self.multi_mask)[1:])    
            self.csv_data.to_csv(self.csv_file_name,index=False)
            self.image_counter += 1
            self.counter= self.image_counter 
            
            
        except Exception as e:
            print(f"Save image error: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save: {str(e)}")

    def masks_refine(self, value):
        if self.raw_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        try:
            if self.SAMBox.isChecked():
                if not self.fastSAM_loaded:
                    self.fastSAM = FastSAM("/SAM-models/FastSAM-s.pt")
                    self.fastSAM_loaded = True
                    self.masks = self.fastSAM_predict()
                else:
                    self.masks = self.fastSAM_predict()
                    
            else:
                self.masks = detect_masks(self.raw_image,self.sigma_value, self.threshold_value)

            self.mask_images, self.multi_mask = show_masks_on_image(self.raw_image, self.masks)
            options = [f"Mask-{i+1}" for i in range(len(self.mask_images))]
            self.maskSelectBox.clear()
            self.maskSelectBox.addItems(options)
            QMessageBox.information(self, "Success", f"{len(self.masks)} masks refined")
        
        except Exception as e:
            print(f"Mask refinement error: {e}")
            QMessageBox.critical(self, "Mask Refinement Error", f"Failed to refine masks: {str(e)}")
    
    def fastSAM_predict(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outputs = self.fastSAM(self.input_name_list[self.image_counter], 
                                device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        outputs = outputs[0].masks.data.cpu().numpy()
        masks = [seg for seg in outputs]
        return masks
    
    def LoadNext(self):
        try:
            if hasattr(self, 'input_name_list') and self.input_name_list:
                if self.image_counter < len(self.input_name_list) - 1:
                    self.maskSelectBox.clear()
                    scene = self.intermView.scene()
                    if scene:
                        scene.clear()
                    
                    self.image_counter += 1
                    self.load_input_image()
                    self.isMaskApplied = False
                    self.selected_mask = None
                    self.masked_image = None
                    self.masks = []
                    self.mask_images = []
                    self.isClassified = False
                    self.curr_class_id = 0
                    self.curr_probability = 0.0
                    self.display_folderInfo()
                    self.ClassSelectBox.setCurrentIndex(0)
                    self.Prob_label.setText(f"Prob: {0.0}")
                    self.ClassId_label.setText(f"Class ID: {0}")
                else:
                    QMessageBox.information(self, "Info", "Last image reached")
                    
        except Exception as e:
            print(f"Load next error: {e}")
    
    def LoadPrevious(self):
        try:
            if hasattr(self, 'input_name_list') and self.input_name_list:
                if self.image_counter < len(self.input_name_list) - 1:
                    self.maskSelectBox.clear()
                    scene = self.intermView.scene()
                    if scene:
                        scene.clear()
                    
                    self.image_counter = self.image_counter - 1 if self.image_counter > 0 else 0
                    self.load_input_image()
                    self.isMaskApplied = False
                    self.selected_mask = None
                    self.masked_image = None
                    self.masks = []
                    self.mask_images = []
                    self.isClassified = False
                    self.curr_class_id = 0
                    self.curr_probability = 0.0
                    self.display_folderInfo()
                    self.ClassSelectBox.setCurrentIndex(0)
                    self.Prob_label.setText(f"Prob: {0.0}")
                    self.ClassId_label.setText(f"Class ID: {0}")
                else:
                    QMessageBox.information(self, "Info", "Last image reached")
                    
        except Exception as e:
            print(f"Load next error: {e}")
    
    def see_class_probabilities(self,value):
        """Display class probabilities in a message box"""
        if self.isClassified:
            classes = list(MaskProcess.color2class.items())
            self.curr_class_id = classes[value][1][1]
            self.curr_probability = self.class_probabilities[value]
            self.Prob_label.setText(f"Prob: {self.class_probabilities[value]:.2f}")
            self.ClassId_label.setText(f"Class ID: {self.curr_class_id}")
            self.display_folderInfo()
        else:
            QMessageBox.warning(self, "Warning", "Please classify the image first")
       
def main():
    """Main function with proper error handling"""
    try:
        print("Creating QApplication...")
        app = QtWidgets.QApplication(sys.argv)
        print("✓ QApplication created")
        
        print("Creating MyApp...")
        window = MyApp()
        print("✓ MyApp created")
        
        print("Showing window...")
        window.show()
        print("✓ Window shown")
        
        print("Starting event loop...")
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"✗ Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()