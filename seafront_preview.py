#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.widgets import Button, RadioButtons, Slider

class SeaFrontPreview:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.current_task = "damage"  # Set default to damage
        self.current_idx = 0
        self.images = []
        self.ann_paths = []
        self.bb_paths = []
        self.tasks = ["damage", "imdg", "door_classification", "ocr"]
        
        # IMDG class mapping
        self.imdg_classes = {
            0: "text", 1: "1-1-1-2-1-3-explosives", 2: "1-4-explosives", 3: "1-5-explosives",
            4: "1-6-explosives", 5: "2-1-a-flammable-gases", 6: "2-1-flammable-gases",
            7: "2-2a-non-flammable-non-toxic-gases", 8: "2-2-non-flammable-non-toxic-gases",
            9: "2-3-toxic-gases", 10: "3a-flammable-liquids", 11: "3-flammable-liquids",
            12: "4-1-flammable-solids-selfreactive-substances-and-solid-desensitized-explosives",
            13: "4-2-substances-liable-to-spontaneous-combustion",
            14: "4-3a-substances-which-in-contact-with-water-emit-flammable-gases-4",
            15: "4-3-substances-which-in-contact-with-water-emit-flammable-gases",
            16: "5-1-oxidizing-substances", 17: "5-2a-organic-peroxides",
            18: "5-2b-organic-peroxides", 19: "6-1-toxic-substances",
            20: "6-2-infectious-substances", 21: "7a-radioactive-material",
            22: "7b-radioactive-material", 23: "7d-radioactive-material",
            24: "7-radioactive-material", 25: "8-corrosive-substances",
            26: "9-miscellaneous-dangerous-substances-and-articles", 27: "container"
        }
        
        # Damage class mapping
        self.damage_classes = {
            0: "container", 1: "axis", 2: "concave", 3: "dentado", 4: "perforation"
        }
        
        self.load_dataset()
        self.setup_ui()
        
    def load_dataset(self):
        print(f"Dataset path: {self.dataset_path}")
        try:
            print(f"Directories in dataset path: {os.listdir(self.dataset_path)}")
            if os.path.exists(os.path.join(self.dataset_path, 'SeaFront_v1_0_0')):
                print(f"Directories in SeaFront_v1_0_0: {os.listdir(os.path.join(self.dataset_path, 'SeaFront_v1_0_0'))}")
        except Exception as e:
            print(f"Error listing directories: {e}")
        
        """Load dataset based on current task"""
        self.images = []
        self.ann_paths = []
        self.bb_paths = []
        
        # Base path to SeaFront dataset
        seafront_path = os.path.join(self.dataset_path, "SeaFront_v1_0_0")
        
        if self.current_task == "damage":
            image_dir = os.path.join(seafront_path, "images", "train")
            ann_dir = os.path.join(seafront_path, "ann_dir", "train")
            bb_dir = os.path.join(seafront_path, "bbannotation")
            
            print(f"Damage - Image dir: {image_dir}, exists: {os.path.exists(image_dir)}")
            print(f"Damage - Ann dir: {ann_dir}, exists: {os.path.exists(ann_dir)}")
            print(f"Damage - BB dir: {bb_dir}, exists: {os.path.exists(bb_dir)}")
            
            if os.path.exists(image_dir) and os.path.exists(ann_dir):
                image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
                
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    # Check if the same filename exists in the annotations directory
                    ann_path = os.path.join(ann_dir, img_file)
                    bb_path = os.path.join(bb_dir, os.path.splitext(img_file)[0] + '.txt') if os.path.exists(bb_dir) else None
                    
                    if os.path.exists(img_path):
                        self.images.append(img_path)
                        self.ann_paths.append(ann_path if os.path.exists(ann_path) else None)
                        self.bb_paths.append(bb_path if bb_path and os.path.exists(bb_path) else None)
            else:
                print("Could not find damage dataset directories")
        
        elif self.current_task == "imdg":
            # First try standard location
            image_dir = os.path.join(seafront_path, "images", "train")
            label_dir = os.path.join(seafront_path, "labels", "train")
            
            # If not found, try detection-specific location
            if not (os.path.exists(image_dir) and os.path.exists(label_dir)):
                image_dir = os.path.join(seafront_path, "ds", "detection", "images", "train")
                label_dir = os.path.join(seafront_path, "ds", "detection", "labels", "train")
            
            print(f"IMDG - Image dir: {image_dir}, exists: {os.path.exists(image_dir)}")
            print(f"IMDG - Label dir: {label_dir}, exists: {os.path.exists(label_dir)}")
            
            if os.path.exists(image_dir) and os.path.exists(label_dir):
                image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
                
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    base_name = os.path.splitext(img_file)[0]
                    label_path = os.path.join(label_dir, base_name + '.txt')
                    
                    if os.path.exists(label_path) and os.path.exists(img_path):
                        self.images.append(img_path)
                        self.ann_paths.append(label_path)
            else:
                print("Could not find IMDG dataset directories")
        
        elif self.current_task == "door_classification":
            door_dir = os.path.join(seafront_path, "ds", "classification", "door")
            no_door_dir = os.path.join(seafront_path, "ds", "classification", "nodoor")
            
            print(f"Door - Door dir: {door_dir}, exists: {os.path.exists(door_dir)}")
            print(f"Door - No door dir: {no_door_dir}, exists: {os.path.exists(no_door_dir)}")
            
            if os.path.exists(door_dir) and os.path.exists(no_door_dir):
                door_files = sorted([os.path.join(door_dir, f) for f in os.listdir(door_dir) 
                                    if f.endswith(('.png', '.jpg'))])
                no_door_files = sorted([os.path.join(no_door_dir, f) for f in os.listdir(no_door_dir) 
                                        if f.endswith(('.png', '.jpg'))])
                
                self.images = door_files + no_door_files
                # Create simple labels: 1 for door, 0 for no door
                self.ann_paths = [1] * len(door_files) + [0] * len(no_door_files)
            else:
                print("Could not find door classification dataset directories")
        
        elif self.current_task == "ocr":
            ocr_dir = os.path.join(seafront_path, "ocr")
            ocr_ann_file = os.path.join(ocr_dir, "ocr_annotations.txt")
            
            print(f"OCR - OCR dir: {ocr_dir}, exists: {os.path.exists(ocr_dir)}")
            print(f"OCR - OCR ann file: {ocr_ann_file}, exists: {os.path.exists(ocr_ann_file)}")
            
            if os.path.exists(ocr_dir) and os.path.exists(ocr_ann_file):
                # Read annotations
                ocr_annotations = {}
                with open(ocr_ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            filename = parts[0]
                            text = ' '.join(parts[1:])
                            ocr_annotations[filename] = text
                
                # Get image files
                image_files = sorted([f for f in os.listdir(ocr_dir) if f.endswith(('.png', '.jpg'))])
                
                for img_file in image_files:
                    img_path = os.path.join(ocr_dir, img_file)
                    self.images.append(img_path)
                    self.ann_paths.append(ocr_annotations.get(img_file, "No annotation"))
            else:
                print("Could not find OCR dataset directories")
        
        if not self.images:
            print(f"No images found for task: {self.current_task}")
            return
            
        self.current_idx = 0
        print(f"Found {len(self.images)} images for task {self.current_task}")
        
    def setup_ui(self):
        """Setup the matplotlib UI"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Add navigation buttons
        prev_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        next_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.prev_button = Button(prev_ax, 'Previous')
        self.next_button = Button(next_ax, 'Next')
        self.prev_button.on_clicked(self.on_prev)
        self.next_button.on_clicked(self.on_next)
        
        # Add task selector
        task_ax = plt.axes([0.1, 0.05, 0.2, 0.15])
        self.task_selector = RadioButtons(task_ax, self.tasks, active=self.tasks.index(self.current_task))
        self.task_selector.on_clicked(self.on_task_change)
        
        # Add image index slider
        slider_ax = plt.axes([0.4, 0.05, 0.2, 0.05])
        self.slider = Slider(slider_ax, 'Image', 0, max(1, len(self.images)-1), valinit=0, valstep=1)
        self.slider.on_changed(self.on_slide)
        
        self.update_display()
        plt.show()
    
    def on_prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.slider.set_val(self.current_idx)
            self.update_display()
    
    def on_next(self, event):
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.slider.set_val(self.current_idx)
            self.update_display()
    
    def on_task_change(self, task):
        self.current_task = task
        self.load_dataset()
        self.slider.set_val(0)
        self.slider.valmax = max(1, len(self.images)-1)
        self.update_display()
    
    def on_slide(self, val):
        self.current_idx = int(val)
        self.update_display()
    
    def update_display(self):
        """Update the image and annotation display"""
        self.ax.clear()
        
        if not self.images:
            self.ax.text(0.5, 0.5, f"No images found for task: {self.current_task}", 
                         ha='center', va='center', fontsize=12)
            self.fig.canvas.draw_idle()
            return
        
        if self.current_idx >= len(self.images):
            self.current_idx = 0
            
        # Display image
        img_path = self.images[self.current_idx]
        img = Image.open(img_path)
        
        # Display appropriate annotations based on task
        if self.current_task == "damage":
            # Display image with segmentation overlay
            self.ax.imshow(img)
            
            # Check if we have a segmentation mask for this image
            if self.current_idx < len(self.ann_paths) and self.ann_paths[self.current_idx] and os.path.exists(self.ann_paths[self.current_idx]):
                # Load segmentation mask
                mask = Image.open(self.ann_paths[self.current_idx])
                mask_np = np.array(mask)
                
                # Create colored overlay for different classes
                overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
                
                # Container is class 0
                container_mask = (mask_np == 0)
                overlay[container_mask] = [0, 255, 0, 100]  # Green with alpha
                
                # Damage classes have values ClassIdx*100+InstanceIdx
                damage_types = {1: [255, 0, 0, 100],    # Axis - Red
                               2: [0, 0, 255, 100],     # Concave - Blue
                               3: [255, 255, 0, 100],   # Dentado - Yellow
                               4: [255, 0, 255, 100]}   # Perforation - Magenta
                
                for class_id, color in damage_types.items():
                    # Find all instances of this class (values starting with class_id*100)
                    instances = np.logical_and(mask_np >= class_id*100, mask_np < (class_id+1)*100)
                    overlay[instances] = color
                
                # Display overlay
                self.ax.imshow(overlay, alpha=0.5)
                
                # Display bounding box info if available
                if self.bb_paths[self.current_idx] and os.path.exists(self.bb_paths[self.current_idx]):
                    with open(self.bb_paths[self.current_idx], 'r') as f:
                        bb_data = f.readlines()
                        
                    for line in bb_data:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1]) * img.width
                            y_center = float(parts[2]) * img.height
                            width = float(parts[3]) * img.width
                            height = float(parts[4]) * img.height
                            
                            # Calculate bbox coordinates
                            x1 = x_center - (width / 2)
                            y1 = y_center - (height / 2)
                            
                            # Draw rectangle
                            rect = plt.Rectangle((x1, y1), width, height, 
                                               linewidth=2, edgecolor='r', facecolor='none')
                            self.ax.add_patch(rect)
                            
                            # Add class label
                            class_name = self.damage_classes.get(cls_id, f"Class {cls_id}")
                            self.ax.text(x1, y1-5, class_name, color='white', 
                                       backgroundcolor='red', fontsize=8)
            else:
                self.ax.set_title("Image without segmentation mask", fontsize=12)
        
        elif self.current_task == "imdg":
            # Display image with IMDG detection boxes
            self.ax.imshow(img)
            
            if self.current_idx < len(self.ann_paths) and os.path.exists(self.ann_paths[self.current_idx]):
                with open(self.ann_paths[self.current_idx], 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1]) * img.width
                        y_center = float(parts[2]) * img.height
                        width = float(parts[3]) * img.width
                        height = float(parts[4]) * img.height
                        
                        # Calculate bbox coordinates
                        x1 = x_center - (width / 2)
                        y1 = y_center - (height / 2)
                        
                        # Draw rectangle
                        rect = plt.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor='r', facecolor='none')
                        self.ax.add_patch(rect)
                        
                        # Add class label
                        class_name = self.imdg_classes.get(cls_id, f"Class {cls_id}")
                        # Truncate long class names
                        if len(class_name) > 20:
                            class_name = class_name[:17] + "..."
                        self.ax.text(x1, y1-5, class_name, color='white', 
                                   backgroundcolor='red', fontsize=8)
        
        elif self.current_task == "door_classification":
            # Display image with door/no door label
            self.ax.imshow(img)
            
            # Show classification
            label = "Door" if self.ann_paths[self.current_idx] == 1 else "No Door"
            self.ax.set_title(f"Classification: {label}", fontsize=14)
        
        elif self.current_task == "ocr":
            # Display OCR image with text annotation
            self.ax.imshow(img)
            
            # Show OCR text
            ocr_text = self.ann_paths[self.current_idx]
            self.ax.set_title(f"OCR Text: {ocr_text}", fontsize=12)
        
        # Set common information
        img_name = os.path.basename(img_path)
        self.ax.set_xlabel(f"File: {img_name} | Task: {self.current_task} | Image {self.current_idx+1}/{len(self.images)}")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.fig.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(description='SeaFront Dataset Preview Tool')
    parser.add_argument('dataset_path', type=str, help='Path to the SeaFront dataset directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        sys.exit(1)
    
    # Check if core directories exist
    seafront_path = os.path.join(args.dataset_path, "SeaFront_v1_0_0")
    if not os.path.exists(seafront_path):
        print(f"Warning: SeaFront_v1_0_0 directory not found at {args.dataset_path}")
        print("Make sure you're pointing to the parent directory of the SeaFront_v1_0_0 folder")
    
    app = SeaFrontPreview(args.dataset_path)

if __name__ == "__main__":
    main() 