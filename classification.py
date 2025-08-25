import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import glob
from PIL import Image
from collections import Counter
import json
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms, models
from tqdm import tqdm

# ---------------------- Multi-Network Architecture ----------------------
class MultiNetworkModel(nn.Module):
    def __init__(self, network_name, num_classes, dropout_prob=0.2, internal_dropout_prob=0.3, full_train=True):
        super(MultiNetworkModel, self).__init__()
        self.network_name = network_name.lower()
        self.full_train = full_train
        self.dropout_prob = dropout_prob
        self.internal_dropout_prob = internal_dropout_prob
        
        # Initialize the backbone based on network choice
        self.backbone = self._get_backbone()
        
        # Setup the model based on architecture
        self._setup_model(num_classes)
        
        # Freeze/unfreeze parameters
        self._setup_training_mode()
    
    def _get_backbone(self):
        """Get the backbone network based on the selected architecture"""
        if self.network_name == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.network_name == 'efficientnet':
            return models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        elif self.network_name == 'convnext':
            return models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        elif self.network_name == 'mobilenet':
            #return models.mobilenet_v3(weights=models.MobileNet_V3_Weights.IMAGENET1K_V2)
            return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        elif self.network_name == 'densenet':
            return models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        elif self.network_name == 'vgg':
            return models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported network: {self.network_name}")
    
    def _setup_model(self, num_classes):
        """Setup the model architecture and classifier head"""
        if self.network_name == 'resnet50':
            self._setup_resnet(num_classes)
        elif self.network_name == 'efficientnet':
            self._setup_efficientnet(num_classes)
        elif self.network_name == 'convnext':
            self._setup_convnext(num_classes)
        elif self.network_name == 'mobilenet':
            self._setup_mobilenet(num_classes)
        elif self.network_name == 'densenet':
            self._setup_densenet(num_classes)
        elif self.network_name == 'vgg':
            self._setup_vgg(num_classes)
    
    def _setup_resnet(self, num_classes):
        """Setup ResNet50 classifier with stable architecture"""
        in_features = self.backbone.fc.in_features
        
        # Replace classifier with stable architecture
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def _setup_efficientnet(self, num_classes):
        """Setup EfficientNet classifier with proper pooling"""
        # EfficientNet already has AdaptiveAvgPool2d in its features
        # The classifier expects flattened features
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier - EfficientNet already pools in features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def _setup_convnext(self, num_classes):
        """Setup ConvNeXt classifier (already stable)"""
        in_features = self.backbone.classifier[2].in_features
        
        # ConvNeXt already has good architecture, just modify final layers
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.LayerNorm((in_features,), eps=1e-06, elementwise_affine=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def _setup_mobilenet(self, num_classes):

        #import pdb
        #pdb.set_trace()
        """Setup MobileNet classifier"""
        in_features = self.backbone.classifier[0].in_features
        
        # MobileNet already has AdaptiveAvgPool2d in features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, num_classes)
        )
    
   
    def _setup_densenet(self, num_classes):
        """Setup DenseNet classifier with proper pooling"""
        # DenseNet already has AdaptiveAvgPool2d in its features
        # The classifier expects flattened features
        in_features = self.backbone.classifier.in_features
        
        # Replace classifier - DenseNet already pools in features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def _setup_vgg(self, num_classes):
        """Setup VGG classifier"""
        # VGG has a more complex classifier structure
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(1024, num_classes)
        )
    
    def _setup_training_mode(self):
        """Setup training mode - freeze/unfreeze parameters"""
        if not self.full_train:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier parameters
            if self.network_name == 'resnet50':
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            else:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
    
    def unfreeze_layers(self, num_layers=1):
        """Gradually unfreeze layers from the end"""
        if self.network_name == 'resnet50':
            layers = ['layer4', 'layer3', 'layer2', 'layer1']
        elif self.network_name == 'efficientnet':
            layers = [f'features.{i}' for i in range(7, 0, -1)]
        elif self.network_name == 'convnext':
            layers = [f'features.{i}' for i in range(7, 0, -1)]
        elif self.network_name == 'mobilenet':
            layers = [f'features.{i}' for i in range(17, 0, -1)]
        elif self.network_name == 'densenet':
            layers = ['features.denseblock4', 'features.denseblock3', 'features.denseblock2', 'features.denseblock1']
        elif self.network_name == 'vgg':
            layers = [f'features.{i}' for i in range(42, 0, -1)]
        else:
            return 0
        
        unfrozen_count = 0
        for layer_name in layers[:num_layers]:
            try:
                layer = self.backbone
                for attr in layer_name.split('.'):
                    layer = getattr(layer, attr)
                
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen_count += 1
                print(f"Unfroze layer: {layer_name}")
            except AttributeError:
                print(f"Warning: Could not find layer {layer_name}")
                continue
        
        return unfrozen_count
    
    def forward(self, x):
        return self.backbone(x)

class MaskProcess:
    # Class-level constant mappings
    color2class = {
       "background":                        [(0,0,0),0],
       # "blank-blank":                      [(255, 255, 255), 1],
       # "blank-oscillatory sector op":      [(255, 0, 0), 2],
        "blank-oscillatory":                [(255, 128, 0), 1],
        #"oscillatory-blank":                [(255, 255, 0), 4],
        #"oscillatory-oscillatory":          [(128, 255, 0), 5],
        "oscillatory-sector op":            [(0, 255, 0), 2],
        #"sector-blank":                     [(0, 255, 128), 7],
        "sector-oscillatory sector op":     [(128, 255, 255), 3],
        #"sector-oscillatory":               [(0, 128, 255), 9],
        "sector-sector":                    [(0, 0, 255), 4],
        #"stripy-oscillatory":               [(128, 0, 255), 11],
        "stripy-stripy":                    [(255, 0, 255), 5],
        #"stripy-blank":                     [(255, 0, 128), 13],
        "wavy-wavy":                        [(255, 128, 128), 6],
        "unclassifiable":                   [(255, 255, 128), 7],
        "metamict":                         [(192, 192, 192), 8],
        # "unsure":                          [(64, 64, 255),17],      
    }

    rgb2class_map = {tuple(v[0]): v[1] for v in color2class.values()}
    class2rgb_map = {v[1]: tuple(v[0]) for v in color2class.values()}
    

    @staticmethod
    def rgb2class(rgbmask):
        classmask = np.zeros((rgbmask.shape[0], rgbmask.shape[1]), dtype=np.uint8)
        for rgb, class_idx in MaskProcess.rgb2class_map.items():
            matches = np.all(rgbmask == rgb, axis=-1)
            matches = np.argwhere(matches)
            if matches.size>0:
                r,c = matches[:,0], matches[:,1]
                classmask[r,c] = class_idx
            else:
                continue
        return classmask

    @staticmethod
    def class2rgb(classmask):
        if classmask.ndim==3:
            rgbmask = np.zeros((classmask.shape[0],classmask.shape[1], classmask.shape[2], 3), dtype=np.uint8)
            for mask,rgb_mask in zip(classmask,rgbmask):
                for class_id, rgb in MaskProcess.class2rgb_map.items():
                    rgb_mask[mask == class_id] = rgb
                
        else:
            rgbmask = np.zeros((classmask.shape[0], classmask.shape[1], 3), dtype=np.uint8)
            for class_id, rgb in MaskProcess.class2rgb_map.items():
                rgbmask[classmask == class_id] = rgb
        return rgbmask

def load_normalization_params(norm_params_file):
    """Load normalization parameters from file."""
    try:
        import json
        with open(norm_params_file, 'r') as f:
            norm_params = json.load(f)
        
        mean = norm_params['mean']
        std = norm_params['std']
        
        print(f"Loaded normalization parameters from {norm_params_file}")
        print(f"Mean: {mean}")
        print(f"Std: {std}")
        print(f"Based on {norm_params.get('num_images', 'unknown')} images")
        
        return mean, std
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Normalization parameters file not found: {norm_params_file}")
    except Exception as e:
        raise ValueError(f"Error loading normalization parameters: {e}")
