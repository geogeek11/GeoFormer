import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import Resize, Normalize
import geopandas as gpd
import mercantile
from rasterio.transform import from_bounds
import logging
import cv2
import pyproj
from rasterio.crs import CRS

class GeoJSONDataset(Dataset):
    """
    Dataset loader for image chips with corresponding GeoJSON labels
    Handles coordinate transformation from Web Mercator to local image coordinates
    """
    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=256,
        start_token=1,
        eos_token=2,
        pad_token=0,
        num_spc_tokens=3,
        sort_polygons=True,
        normalize_ims=True,
        **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
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
        
        # Initialize transformers for coordinate conversion
        self.web_to_wgs84 = pyproj.Transformer.from_crs(
            CRS.from_epsg(3857),  # Web Mercator
            CRS.from_epsg(4326),  # WGS84
            always_xy=True
        )
        
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

    def get_tile_bounds(self, x, y, z):
        """Get tile bounds in Web Mercator coordinates"""
        tile = mercantile.Tile(x=x, y=y, z=z)
        bounds = mercantile.bounds(tile)
        # Convert to Web Mercator
        west, south = mercantile.xy(bounds.west, bounds.south)
        east, north = mercantile.xy(bounds.east, bounds.north)
        return west, south, east, north

    def transform_coordinates(self, coords, west, south, east, north, image_size):
        """
        Transform coordinates from Web Mercator to local image coordinates
        Args:
            coords: numpy array of shape (N, 2) containing Web Mercator coordinates
            west, south, east, north: tile bounds in Web Mercator
            image_size: size of the output image
        Returns:
            numpy array of shape (N, 2) containing local image coordinates
        """
        # Scale coordinates to image space
        x_scale = image_size / (east - west)
        y_scale = image_size / (north - south)
        
        # Transform coordinates
        x = (coords[:, 0] - west) * x_scale
        y = image_size - (coords[:, 1] - south) * y_scale  # Flip y-axis
        
        return np.stack([x, y], axis=1)

    def poly_to_mask(self, vertices, image_size):
        """Convert polygon vertices to binary mask"""
        img = np.zeros((image_size, image_size), np.uint8)
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(img, [vertices], 1)
        mask = np.clip(img, 0, 1).astype(np.uint8)
        return mask

    def load_geojson(self, label_path, tile_x, tile_y, tile_z):
        """
        Load and process GeoJSON file
        Args:
            label_path: path to GeoJSON file
            tile_x, tile_y, tile_z: XYZ tile coordinates
        """
        try:
            # Get tile bounds in Web Mercator
            west, south, east, north = self.get_tile_bounds(tile_x, tile_y, tile_z)
            
            # Load GeoJSON file
            gdf = gpd.read_file(label_path)
            if gdf.crs != 'EPSG:3857':
                gdf = gdf.to_crs('EPSG:3857')
            
            polygons = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                # Extract coordinates from the geometry
                if geom.geom_type == 'Polygon':
                    polys = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    polys = list(geom.geoms)
                else:
                    continue
                    
                for poly in polys:
                    try:
                        # Get coordinates in Web Mercator
                        coords = np.array(poly.exterior.coords)
                        if len(coords) < 3:  # Skip invalid polygons
                            continue
                            
                        # Transform to local image coordinates
                        transformed_coords = self.transform_coordinates(
                            coords, west, south, east, north, self.image_size
                        )
                        
                        # Add padding to prevent edge artifacts
                        padding = 0
                        transformed_coords = np.clip(
                            transformed_coords,
                            padding,
                            self.image_size - padding
                        )
                        
                        polygons.append(transformed_coords)
                        
                    except (ValueError, AttributeError) as e:
                        self.logger.warning(f"Skipping invalid polygon: {e}")
                        continue
            
            if self.sort_polygons and polygons:
                from src.utils.utils import sort_polygons
                polygons = sort_polygons(polygons, return_indices=False, im_dim=self.image_size)
            
            return polygons
            
        except Exception as e:
            self.logger.error(f"Error loading GeoJSON file {label_path}: {e}")
            return []

    def __getitem__(self, idx):
        try:
            # Load image
            image_file = self.image_files[idx]
            image_path = os.path.join(self.image_dir, image_file)
            
            # Extract tile coordinates from filename (format: OAM-{ID}-{X}-{Y}-{Z}.jpg)
            # Example: OAM-1861240-826122-21.jpg
            try:
                parts = os.path.splitext(image_file)[0].split('-')
                if len(parts) < 4:
                    raise ValueError(f"Invalid filename format: {image_file}")
                # Extract coordinates from the end of the filename
                z = int(parts[-1])  # Last part is zoom level
                y = int(parts[-2])  # Second to last is Y
                x = int(parts[-3])  # Third to last is X
            except (ValueError, IndexError) as e:
                self.logger.error(f"Error parsing filename {image_file}: {e}")
                raise
            
            # Get corresponding label file, maintaining the OAM identifier format
            label_file = os.path.splitext(image_file)[0] + '.geojson'
            label_path = os.path.join(self.label_dir, label_file)
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image = Resize(self.output_size)(image)
            image = F.to_tensor(image)
            
            if self.normalizer:
                image = self.normalizer(image)
            
            # Load and process polygons with correct coordinate transformation
            polygons = self.load_geojson(label_path, x, y, z)
            
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
            
            # Handle case with no valid polygons
            if not processed_polygons:
                dummy_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                return {
                    "image": image,
                    "vtx_list": [],
                    "vertices_flat": [],
                    "mask": torch.from_numpy(dummy_mask),
                    "img_id": image_file
                }
            
            # Combine masks
            combined_mask = np.max(np.stack(masks, axis=0), axis=0)
            
            return {
                "image": image,
                "vtx_list": processed_polygons,
                "vertices_flat": [],  # Will be processed by collate_fn
                "mask": torch.from_numpy(combined_mask).float(),
                "img_id": image_file
            }
            
        except Exception as e:
            self.logger.error(f"Error processing index {idx} (file: {self.image_files[idx]}): {e}")
            raise