import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry,SamPredictor
from ultralytics import FastSAM
from skimage.measure import label, regionprops
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
from PyQt6 import QtWidgets, uic,QtCore
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog,QMessageBox,QGraphicsScene, QGridLayout
from PyQt6.QtCore import QTimer
import glob
import os
from PIL import Image,ImageQt
from PIL.ImageQt import toqimage,toqpixmap
from utility import show_masks_on_image, masks_filter
import time
import cv2
import pandas as pd
import json
# Load the UI file
ui_file = "Particle-Segment.ui"  # Path to the .ui file

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi(ui_file, self)  # Load the .ui file and setup the UI
        self.initUI()
        self.sam = None#sam_model_registry["vit_b"](checkpoint=r"/media/nazib/Nazibs/Correlation_Analysis_data/code/SAM-models/sam_vit_b_01ec64.pth")
        self.fastSAM = None#FastSAM("FastSAM-s.pt") 
        '''
        self.mask_generator = SamAutomaticMaskGenerator(self.sam,
                            points_per_side=8,points_per_batch=16)
        '''
        columns = ['FileIndex','FileName','Time','MasksDetected',
                   'MasksFiltered','MasksPreserved','SavedImages','SavedLabels']
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
            self.output_folder_path=self.csv_data.attrs['OutputFolderPath']
            self.out_dir_box.setText(self.output_folder_path)
            self.input_name_list = self.csv_data.attrs['InputNameList']
            self.display_folderInfo()
            QTimer.singleShot(0, self.load_input_image)
        else:
            self.input_dir =""
            self.dict ={}
            self.image_counter = 0

    def initUI(self):
        self.detected_contours =0
        '''
        self.gridLayout.addWidget(self.inputView)
        self.gridLayout.addWidget(self.input_dir_box)
        self.gridLayout.addWidget(self.output_dir_box)
        self.setLayout(self.gridLayout)
        '''
        # You can connect signals to slots here if needed
        self.SelectMain_btn.clicked.connect(self.select_input_folder_dialog)
        self.ExitMain_btn.clicked.connect(self.close_application)
        self.SelectOutput_btn.clicked.connect(self.select_output_folder_dialog)
        self.Segment_btn.clicked.connect(self.segmentation)

        #self.intermSlider.setMaximum(self.detected_contours)
        #self.intermSlider.valueChanged.connect(self.scroll_image)
        self.DiffSelectBox.addItems(['1','2','3','4','5'])
        self.maskSelectBox.addItems([f"Mask-{0}",f"Mask-{0}",f"Mask-{0}"]) 
        self.maskSelectBox.value=0
        self.modelSelectBox.addItems(["SAM","SAM2","FastSAM"])
        self.modelSelectBox.value=0
        self.SamParamBox.addItems(['4','8','16','32'])
        self.SamParamBox.value = 0
        # Connect the QComboBox to an event handler
        self.maskSelectBox.currentIndexChanged.connect(self.scroll_image)
        ### Save Image button event handler ###
        self.SaveImage_btn.clicked.connect(self.SaveImage)
        ### Load Next image image button ###
        self.LoadNext_btn.clicked.connect(self.LoadNext) 

    def close_application(self):
        self.close()

    def select_input_folder_dialog(self):
        # Open a dialog to select a folder
        file_ext = ["*.tif","*.jpg","*.png","*.jpeg","*.JPEG","*.PNG"]
        #folder_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.tif)")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

        if folder_path:
            self.input_dir = folder_path
            self.input_dir_box.setText(folder_path)
            for ext in file_ext:
                self.input_name_list = glob.glob(os.path.join(self.input_dir,ext))
                if self.input_name_list:
                    self.input_type = ext
                    break
                else:
                    continue
            if not self.input_name_list:
                QMessageBox.warning(self,"No images are found","Supported formets are tif jpg png jpeg JPEG PNG")
        else:
            QMessageBox.warning(self,"Invalid Folder","Please ensure directory")
        self.load_input_image()
        self.csv_data.attrs["InputFolderPath"] = folder_path
        self.csv_data.attrs["InputNameList"] = self.input_name_list
    
    def display_folderInfo(self):
        self.info = (
        f"Input Dir: {os.path.basename(self.input_dir)}\n"
        f"Output Dir: {os.path.basename(self.output_folder_path)}\n"
        f"Total: {len(self.input_name_list)} Images\n"
        f"Image: {os.path.basename(self.input_name_list[self.image_counter])}\n"
        f"Processed:{self.image_counter}\n"
        )
        self.FolderInfo.setText(self.info)
        self.dict['FileName'] = os.path.basename(self.input_name_list[self.image_counter])
        self.dict['FileIndex'] = self.image_counter

    def load_image_on_startup(self):
        pixmap = QPixmap(self.input_name_list[self.image_counter])
        # Create a QGraphicsScene and add the pixmap to it
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        # Set the scene to the QGraphicsView (inputView widget)
        self.ui.inputView.setScene(scene)
        # Adjust the view to fit the image
        self.ui.inputView.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def load_input_image(self):
        # Load the image into a QPixmap
        pixmap = QPixmap(self.input_name_list[self.image_counter])
        # Create a QGraphicsScene and add the QPixmap to it
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        # Set the scene to the QGraphicsView
        self.inputView.setScene(scene)
        self.inputView.scale(2, 2)
        #self.inputView.fitInView(scene.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
        self.inputView.fitInView(scene.itemsBoundingRect())
    def segmentation(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.raw_image = Image.open(self.input_name_list[self.image_counter]).convert("RGB")

        if self.modelSelectBox.currentText() == "SAM":
            number_of_points = int(self.SamParamBox.currentText())
            self.mask_generator = SamAutomaticMaskGenerator(self.sam,
                    points_per_side= number_of_points,points_per_batch=2*number_of_points)
            #### Original SAM #####
            st= time.time()
            outputs = self.mask_generator.generate(np.asarray(self.raw_image))
            en = time.time()
            ex_time = en-st
            self.masks = masks_filter(self.raw_image,outputs)
        elif self.modelSelectBox.currentText() == "SAM2":
            predictor = SAM2ImagePredictor(build_sam2(r"C:\BHP\Correlation_Analysis_data\code\SAM-models\sam2_hiera_tiny.yaml", 
                                                      r"C:\BHP\Correlation_Analysis_data\code\SAM-models\sam2_hiera_tiny.pt"))
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                predictor.set_image(np.asarray(self.raw_image))
                outputs, _, _ = predictor.predict()
        else:
            #### Original FastSAM #####
            st= time.time()
            outputs = self.fastSAM(self.input_name_list[self.image_counter], 
                                device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            outputs = outputs[0].masks.data.numpy()
            self.masks = [seg for seg in outputs]
            en = time.time()
            ex_time = en-st
        
        self.mask_images,self.multi_mask = show_masks_on_image(self.raw_image, self.masks)
        self.detected_contours = len(self.mask_images)
        #self.intermSlider.setMaximum(self.detected_contours)
        options = [f"Mask-{i+1}" for i in range(len(self.mask_images))]
        self.maskSelectBox.clear()
        self.maskSelectBox.addItems(options)
        status = (f"Masks Detected:{len(outputs)}\n"
                  f"Detection Time:{ex_time:.2f}s\n"
                  f"Masks Filtered:{len(outputs)-len(self.masks)}\n"
                  f"File Count:{self.image_counter+1}\n"
                  f"Remaining Images:{len(self.input_name_list)-(self.image_counter+1)}"
                  )
        self.display_folderInfo()
        self.info+= status
        self.FolderInfo.setText(self.info)
        self.ImageName.setText(os.path.basename(self.input_name_list[self.image_counter]))
        self.dict['Time'] = ex_time
        self.dict['MasksDetected'] = len(outputs)
        self.dict['MasksFiltered'] = len(outputs) - len(self.masks)
        self.dict['MasksPreserved'] = len(self.masks)
        
        
    def scroll_image(self,value):
        if value>=0 and value<len(self.masks):
            self.selected_mask = self.masks[value]
            self.counter = value
        elif value==len(self.masks)-1:
            self.selected_mask = self.masks[value-1]
            self.counter = value-1
        else:
             value = 0
             self.selected_mask = self.masks[value]
             self.counter = value

        image_array = self.mask_images[value]
        image_array = Image.fromarray(image_array,"RGB")
        qimage = ImageQt.toqimage(image_array)
        #Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qimage)
        # Create a QGraphicsScene and add the QPixmap to it
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        # Set the scene to the QGraphicsView
        self.intermView.setScene(scene)
        self.intermView.fitInView(scene.itemsBoundingRect())

    def select_output_folder_dialog(self):
        #Open a dialog to select a folder
        self.output_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.output_folder_path:
            self.out_dir_box.setText(self.output_folder_path)
            self.display_folderInfo()
        self.csv_data.attrs['OutputFolderPath']=self.output_folder_path
        self.csv_data.to_csv(self.csv_file_name,index=False)
        with open('metadata.json', 'w') as meta_file:
            json.dump(self.csv_data.attrs, meta_file)

    def SaveImage(self):
        src_image = np.asarray(self.raw_image).copy()
        region = regionprops(label(self.selected_mask))
        cut_out = np.zeros(shape=src_image.shape,dtype=src_image.dtype)
        cut_out[:,:] = src_image.min()
        x,y,w,h = region[0].bbox
        cut_out[x:x+w,y:y+h] = src_image[x:x+w,y:y+h]
        cut_out[self.selected_mask==0] = src_image.min()
        file_name_parts = os.path.basename(self.input_name_list[self.image_counter]).split(".")
        file_name = f"{file_name_parts[0]}_{self.counter+1}.{file_name_parts[1]}"
        file_name_mask = f"{file_name_parts[0]}_mask.{file_name_parts[1]}"
        save_path =os.path.join(self.output_folder_path,file_name)
        cv2.imwrite(save_path,cut_out)
        save_path =os.path.join(self.output_folder_path,'Masks')

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imwrite(os.path.join(save_path,file_name_mask),self.multi_mask)
        
        if self.counter ==0 or len(self.csv_data)==0:
            self.dict["SavedImages"] = file_name
            self.dict["SavedLabels"] = str(np.unique(self.multi_mask)[1:])
            new_data = pd.DataFrame(self.dict,index=[self.image_counter])
            self.csv_data = pd.concat([self.csv_data,new_data],ignore_index=True)
        else:
            df_slice = self.csv_data.iloc[self.image_counter].copy()
            df_slice['SavedImages'] += f",{file_name}"
            df_slice['SavedLabels'] += f",{str(np.unique(self.multi_mask)[1:])}"
            self.csv_data.iloc[self.image_counter]= df_slice
            #self.csv_data['SavedLabels'].loc[self.image_counter]+= str(np.unique(self.multi_mask)[1:])    
        self.csv_data.to_csv(self.csv_file_name,index=False)
        self.counter+=1
        
    
    def LoadNext(self):
        if self.maskSelectBox.itemText(0) == 'Mask-1':
            self.maskSelectBox.clear()
            self.intermView.items().clear()
        
        self.image_counter+=1
        self.load_input_image()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())
    #sys.exit(app.exec_())