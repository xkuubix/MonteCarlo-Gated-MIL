import torch
import numpy as np
from sklearn.utils import shuffle
import logging
logger = logging.getLogger(__name__)

class ImagePatcher:
    def __init__(self, patch_size=224, overlap=0.5, bag_size=-1, empty_thresh=0.8):
        self.patch_size = patch_size
        self.overlap = overlap
        self.bag_size = bag_size
        self.empty_thresh = empty_thresh
        self.tiles = None
        logger.info(f"ImagePatcher initialized with patch_size={patch_size}, overlap={overlap}, bag_size={bag_size}, empty_thresh={empty_thresh}")

    def _start_points(self, size, split_size):
        points = [0]
        stride = int(split_size * (1 - self.overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    def get_tiles(self, h, w):
        X_points = self._start_points(w, self.patch_size)
        Y_points = self._start_points(h, self.patch_size)
        
        tiles = np.zeros((len(Y_points) * len(X_points), 6), int)
        k = 0
        for i, y_cord in enumerate(Y_points):
            for j, x_cord in enumerate(X_points):
                tiles[k] = (y_cord, x_cord, self.patch_size, self.patch_size, i, j)
                k += 1
        self.tiles = tiles
        return tiles

    def convert_img_to_bag(self, image):
        tiles = self.tiles
        hTile, wTile = tiles[0][2], tiles[0][3]
        c = image.shape[0]
        img_shape = (len(tiles), c, hTile, wTile)
        new_img = torch.zeros(img_shape)
        px_non_zero = torch.zeros(len(tiles), dtype=torch.float32)
        
        for i, tile in enumerate(tiles):
            new_img[i] = image[:, tile[0]:tile[0]+tile[2], tile[1]:tile[1]+tile[3]]
            px_non_zero[i] = (new_img[i][0] > 0).float().mean() * 100
        
        sorted_tiles_idx = np.argsort(-px_non_zero)
        px_non_zero_pc = (px_non_zero > self.empty_thresh * 100).sum()
        
        instances, instances_idx, instances_cords = self._select_bag(new_img, tiles, sorted_tiles_idx, px_non_zero_pc)
        return instances, instances_idx, instances_cords
    
    
    def reconstruct_image_from_patches(self, patches, instances_ids, image_shape):
        tiles = self.tiles
        c, h, w = image_shape
        
        # Initialize with zeros
        reconstructed_image = torch.zeros(c, h, w, dtype=patches[0].dtype, device=patches[0].device)
        patch_count = torch.zeros(c, h, w, dtype=torch.float, device=patches[0].device)

        for item in range(len(instances_ids)):
            h_min, w_min, dh, dw, _, _ = tiles[instances_ids[item]]
            patch = patches[item]

            # Accumulate patch contributions
            reconstructed_image[:, h_min:h_min + dh, w_min:w_min + dw] += patch
            patch_count[:, h_min:h_min + dh, w_min:w_min + dw] += 1

        # Avoid division by zero (set to 1 where count is 0)
        patch_count = torch.where(patch_count == 0, torch.ones_like(patch_count), patch_count)

        # Normalize by count to average overlapping regions
        reconstructed_image /= patch_count

        return reconstructed_image
 

    def _select_bag(self, new_img, tiles, sorted_tiles_idx, px_non_zero_pc):        
        
        if self.bag_size > 0:
            if self.bag_size > px_non_zero_pc:
                instances = new_img[sorted_tiles_idx[:px_non_zero_pc]]
                instances_idx = sorted_tiles_idx[:px_non_zero_pc]
            else:
                instances = new_img[sorted_tiles_idx[:self.bag_size]]
                instances_idx = sorted_tiles_idx[:self.bag_size]
        elif self.bag_size == -1:
            instances = new_img[sorted_tiles_idx[:px_non_zero_pc]]
            instances_idx = sorted_tiles_idx[:px_non_zero_pc]
        else:
            raise ValueError("Invalid bag size")
        
        instances_cords = tiles[instances_idx, 4:6]
        return shuffle(instances, instances_idx, instances_cords)
