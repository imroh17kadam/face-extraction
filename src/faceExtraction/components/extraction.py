import os
import sys


project_path = 'C:\\Users\\DELL\\OneDrive\\Desktop\\MachineLearning\\face-extraction'
detectron2_path = os.path.join(project_path, 'detectron2')

# Add the top-level Detectron2 directory to the Python path
sys.path.insert(0, detectron2_path)


# Import Detectron2 modules
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

# Set up the logger
setup_logger()

# Import common libraries
import numpy as np
import json
import cv2
import random

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import torch




class FaceExtractor:
    def __init__(self, score_thresh=0.5, device=None):
        self.model_config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.model_weights = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.model_config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.MODEL.WEIGHTS = self.model_weights
        self.cfg.MODEL.DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = DefaultPredictor(self.cfg)
        self.person_class_id = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes.index('person')
    
    def read_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Unable to read image at {image_path}")
    
    def predict(self):
        self.outputs = self.predictor(self.image)
        self.instances = self.outputs["instances"]
        self.masks = self.instances.pred_masks.to("cpu").numpy()
        self.classes = self.instances.pred_classes.to("cpu").numpy()
    
    def extract_faces(self):
        face_masks = [self.masks[i] for i in range(len(self.masks)) if self.classes[i] == self.person_class_id]
        combined_mask = np.sum(face_masks, axis=0)
        output_image = np.zeros_like(self.image)
        output_image[combined_mask > 0] = self.image[combined_mask > 0]
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
        output_image[combined_mask == 0, 3] = 0
        return output_image
    
    def save_image(self, output_image, output_path):
        cv2.imwrite(output_path, output_image)