import os
os.environ["TORCH_HOME"] = r"D:\model"

import torch
import numpy as np
import torch.nn as nn
from torchvision.models import (resnet18, ResNet18_Weights, resnet50, ResNet50_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights,                                 
                                mobilenet_v3_large, MobileNet_V3_Large_Weights)
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import json

stats_to_class = {
        (0, 0, 0, 0, 0): 0,
        (0, 0, 0, 1, 0): 1,
        (0, 0, 0, 1, 1): 2,
        (0, 1, 0, 1, 0): 3,
        (1, 0, 1, 0, 0): 4,
        (1, 0, 0, 0, 0): 5,
        (1, 0, 0, 0, 1): 6,
        (1, 0, 0, 1, 0): 7,
        (1, 1, 1, 0, 0): 8,
        (1, 1, 0, 0, 0): 9,
        (1, 1, 0, 1, 0): 10,
        (2, 0, 0, 0, 0): 11,
        (2, 1, 0, 0, 0): 12,
        }

class Resnet18_Modified(nn.Module):
    def __init__(self, num_shapes, num_edges, num_textures, num_sizes, num_colours, num_classes):
        super(Resnet18_Modified, self).__init__()

        self.backbone = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create separate classification heads for each attribute
        self.shape_classifier = nn.Linear(num_features, num_shapes)
        self.edge_classifier = nn.Linear(num_features, num_edges)
        self.texture_classifier = nn.Linear(num_features, num_textures)
        self.size_classifier = nn.Linear(num_features, num_sizes)
        self.colour_classifier = nn.Linear(num_features, num_colours)

        # A classifier for the class
        self.class_classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)

        return {
            "class": self.class_classifier(features),
            "shape": self.shape_classifier(features),
            "edge": self.edge_classifier(features),
            "texture": self.texture_classifier(features),
            "size": self.size_classifier(features),
            "colour": self.colour_classifier(features),
        }  
    
class Resnet18(nn.Module):
    def __init__(self, num_shapes, num_edges, num_textures, num_sizes, num_colours):
        super(Resnet18, self).__init__()

        self.backbone = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create separate classification heads for each attribute
        self.shape_classifier = nn.Linear(num_features, num_shapes)
        self.edge_classifier = nn.Linear(num_features, num_edges)
        self.texture_classifier = nn.Linear(num_features, num_textures)
        self.size_classifier = nn.Linear(num_features, num_sizes)
        self.colour_classifier = nn.Linear(num_features, num_colours)

    def forward(self, x):
        features = self.backbone(x)
        shape_output = self.shape_classifier(features)
        edge_output = self.edge_classifier(features)
        texture_output = self.texture_classifier(features)
        size_output = self.size_classifier(features)
        colour_output = self.colour_classifier(features)
        return shape_output, edge_output, texture_output, size_output, colour_output

class MobileNetV3L(nn.Module):
    def __init__(self, num_shapes, num_edges, num_textures, num_sizes, num_colours):
        super(MobileNetV3L, self).__init__()
        self.backbone = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        # Create separate classification heads for each attribute
        self.shape_classifier = nn.Linear(num_features, num_shapes)
        self.edge_classifier = nn.Linear(num_features, num_edges)
        self.texture_classifier = nn.Linear(num_features, num_textures)
        self.size_classifier = nn.Linear(num_features, num_sizes)
        self.colour_classifier = nn.Linear(num_features, num_colours)

    def forward(self, x):
        features = self.backbone(x)
        shape_output = self.shape_classifier(features)
        edge_output = self.edge_classifier(features)
        texture_output = self.texture_classifier(features)
        size_output = self.size_classifier(features)
        colour_output = self.colour_classifier(features)
        return shape_output, edge_output, texture_output, size_output, colour_output

class EfficientNet_B1(nn.Module):
    def __init__(self, num_shapes, num_edges, num_textures, num_sizes, num_colours):
        super(EfficientNet_B1, self).__init__()
        self.backbone = efficientnet_b1(weights = EfficientNet_B1_Weights.IMAGENET1K_V2)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Create separate classification heads for each attribute
        self.shape_classifier = nn.Linear(num_features, num_shapes)
        self.edge_classifier = nn.Linear(num_features, num_edges)
        self.texture_classifier = nn.Linear(num_features, num_textures)
        self.size_classifier = nn.Linear(num_features, num_sizes)
        self.colour_classifier = nn.Linear(num_features, num_colours)

    def forward(self, x):
        features = self.backbone(x)
        shape_output = self.shape_classifier(features)
        edge_output = self.edge_classifier(features)
        texture_output = self.texture_classifier(features)
        size_output = self.size_classifier(features)
        colour_output = self.colour_classifier(features)
        return shape_output, edge_output, texture_output, size_output, colour_output

class EfficientNet_B0(nn.Module):
    def __init__(self, num_shapes, num_edges, num_textures, num_sizes, num_colours):
        super(EfficientNet_B0, self).__init__()
        self.backbone = efficientnet_b0(weights = EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Create separate classification heads for each attribute
        self.shape_classifier = nn.Linear(num_features, num_shapes)
        self.edge_classifier = nn.Linear(num_features, num_edges)
        self.texture_classifier = nn.Linear(num_features, num_textures)
        self.size_classifier = nn.Linear(num_features, num_sizes)
        self.colour_classifier = nn.Linear(num_features, num_colours)

    def forward(self, x):
        features = self.backbone(x)
        shape_output = self.shape_classifier(features)
        edge_output = self.edge_classifier(features)
        texture_output = self.texture_classifier(features)
        size_output = self.size_classifier(features)
        colour_output = self.colour_classifier(features)
        return shape_output, edge_output, texture_output, size_output, colour_output

class Resnet50(nn.Module):
    def __init__(self, num_shapes, num_edges, num_textures, num_sizes, num_colours):
        super(Resnet50, self).__init__()

        self.backbone = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create separate classification heads for each attribute
        self.shape_classifier = nn.Linear(num_features, num_shapes)
        self.edge_classifier = nn.Linear(num_features, num_edges)
        self.texture_classifier = nn.Linear(num_features, num_textures)
        self.size_classifier = nn.Linear(num_features, num_sizes)
        self.colour_classifier = nn.Linear(num_features, num_colours)

    def forward(self, x):
        features = self.backbone(x)
        shape_output = self.shape_classifier(features)
        edge_output = self.edge_classifier(features)
        texture_output = self.texture_classifier(features)
        size_output = self.size_classifier(features)
        colour_output = self.colour_classifier(features)
        return shape_output, edge_output, texture_output, size_output, colour_output




class FungusDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.mapping = class_mapping(root_dir)

        self.image_paths = []
        self.shape_labels = []
        self.edge_labels = []
        self.texture_labels = []
        self.size_labels = []
        self.colour_labels = []
        self.class_labels = []
        
        for class_info in self.mapping:
            folder_path = class_info["folder_path"]
            class_stats = class_info["class_stats"]
            class_number = class_info["class_number"]

            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                self.image_paths.append(img_path)
                
                # Extract class labels
                self.shape_labels.append(class_stats[0])
                self.edge_labels.append(class_stats[1])
                self.texture_labels.append(class_stats[2])
                self.size_labels.append(class_stats[3])
                self.colour_labels.append(class_stats[4])
                self.class_labels.append(class_number)   

         
        print(f"Found {len(self.image_paths)} images across {len(self.mapping)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        image = self.transform(image)
        
        # Return all 5 attribute labels
        labels = (
            self.shape_labels[idx],
            self.edge_labels[idx],
            self.texture_labels[idx],
            self.size_labels[idx],
            self.colour_labels[idx]
        )
        class_label = self.class_labels[idx]
        return image, labels, class_label


def class_mapping(root_dir):
    root_path = Path(root_dir)
    mapping = []

    for class_path in root_path.iterdir():
        if not class_path.is_dir():
            continue
        
        class_name = class_path.name
        parts = class_name.split("、")
        if parts[0] == "不规则形":
            shape = 0
        elif parts[0] == "圆形":
            shape = 1
        elif parts[0] == "环形":
            shape = 2

        if parts[1] == "平滑状":
            edge = 0
        elif parts[1] == "毛刺状":
            edge = 1

        if parts[2] == "绒毛状":
            texture = 0
        elif parts[2] == "粘液状且绒毛状":
            texture = 1

        if parts[3] == "大":
            size = 0
        elif parts[3] == "小":
            size = 1

        if parts[4] == "白色":
            colour = 0
        elif parts[4] == "褐色":
            colour = 1

        class_stats = [shape, edge, texture, size, colour]
        class_number = stats_to_class.get(tuple(class_stats))
        if class_number is None:
            raise ValueError(f"Unknown class combination: {class_stats} for {class_name}")
        mapping.append({
            "class_name": class_name,
            "class_stats": class_stats,
            "class_number": class_number,
            "folder_path": str(class_path)
        })

    with open(root_path / 'detailed_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping
