import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import Resize , Normalize
import geopandas as gpd
from shapely.geometry import Polygon, mapping

class GeoJSONDataset(Dataset):
    """
    Dataset loader for image chips with corresponding GeoJSON labels
    """
    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=224,
        start_token=1,
        eos_token=2,
        pad_token=0,
        num_spc_tokens=3,
        sort_polygons=True,
        normalize_ims=True,
        **kwargs
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.output_size = (image_size, image_size)
        self.start_token = start_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.num_spc_tokens = num_spc_tokens
        self.sort_polygons = sort_polygons
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif'))]
        
        # Normalization parameters
        if normalize_ims:
            self.normalizer = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalizer = None

    def __len__(self):
        return len(self.image_files)

    def poly_to_mask(self, vertices, image_size):
        """Convert polygon vertices to binary mask"""
        img = np.zeros((image_size, image_size), np.uint8)
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(img, [vertices], 1)
        mask = np.clip(img, 0, 1).astype(np.uint8)
        return mask

    def load_geojson(self, label_path):
        """Load and process GeoJSON file"""
        gdf = gpd.read_file(label_path)
        polygons = []
        
        for _, row in gdf.iterrows():
            # Extract coordinates from the geometry
            if row.geometry.geom_type == 'Polygon':
                coords = np.array(row.geometry.exterior.coords)
                # Scale coordinates to image space
                coords = coords * (self.image_size / max(coords.max() - coords.min()))
                # Center the coordinates
                coords = coords - coords.min() + 10  # Add padding of 10 pixels
                polygons.append(coords)
            elif row.geometry.geom_type == 'MultiPolygon':
                for polygon in row.geometry:
                    coords = np.array(polygon.exterior.coords)
                    coords = coords * (self.image_size / max(coords.max() - coords.min()))
                    coords = coords - coords.min() + 10
                    polygons.append(coords)
        
        if self.sort_polygons:
            from src.utils.utils import sort_polygons
            polygons = sort_polygons(polygons, return_indices=False, im_dim=self.image_size)
        
        return polygons

    def __getitem__(self, idx):
        # Load image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.geojson'
        label_path = os.path.join(self.label_dir, label_file)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image = Resize(self.output_size)(image)
        image = F.to_tensor(image)
        
        if self.normalizer:
            image = self.normalizer(image)
        
        # Load and process polygons
        polygons = self.load_geojson(label_path)
        
        # Process vertices for each polygon
        processed_polygons = []
        masks = []
        
        for poly in polygons:
            # Add special tokens offset
            vtx = poly + self.num_spc_tokens
            
            # Create mask
            mask = self.poly_to_mask(poly, self.image_size)
            masks.append(mask)
            
            # Ensure polygon is closed
            if not np.array_equal(vtx[0], vtx[-1]):
                vtx = np.vstack([vtx, vtx[0]])
            
            processed_polygons.append(vtx)
        
        # Combine masks
        combined_mask = np.max(np.stack(masks, axis=0), axis=0)
        
        return {
            "image": image,
            "vtx_list": processed_polygons,
            "vertices_flat": [],  # Will be processed by collate_fn
            "mask": torch.from_numpy(combined_mask).float(),
            "img_id": image_file
        }