from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import cv2
import random

import albumentations as A
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.filters import roberts
from scipy.ndimage import binary_erosion


class NPZSplitSegmentationDataset(Dataset):
	"""Dataset that reads one sample per .npz from a split directory."""

	def __init__(
		self,
		split_dir,
		normalize_image=True,
		binarize_mask=True,
		mask_threshold=0.5,
		apply_edge_maps=True,
		apply_augmentations=False,
		apply_albumentations=False,
		apply_edge_detection=True,
		quadrant_crop_N=1,
	):
		self.split_dir = Path(split_dir)
		self.normalize_image = normalize_image
		self.binarize_mask = binarize_mask
		self.mask_threshold = float(mask_threshold)
		self.apply_augmentations = apply_augmentations
		self.apply_albumentations = apply_albumentations
		self.apply_edge_maps = apply_edge_maps
		self.apply_edge_detection = apply_edge_detection
		self.quadrant_crop_N = quadrant_crop_N
		self.use_quadrant_crop = quadrant_crop_N > 1
		self.npz_files = sorted(self.split_dir.glob("*.npz"))
		if len(self.npz_files) == 0:
			raise ValueError(f"No .npz samples found in split directory: {self.split_dir}")
		
		self.quad_crop = RandomQuadrantCrop(n=quadrant_crop_N)
		self._joint_transform = self._build_joint_transform()
		self._image_only_transform = self._build_image_only_transform()


	def __len__(self):
		return len(self.npz_files) # arbitrary length multiplier

	@staticmethod
	def _imread_zero_alpha(path, as_gray=False):
		arr = imageio.imread(path)

		# RGBA
		if arr.ndim == 3 and arr.shape[2] == 4:
			rgb = arr[..., :3].astype(np.float32)
			alpha = arr[..., 3].astype(np.float32) / 255.0
			rgb[alpha == 0] = 0.0
			rgb *= alpha[..., None]
			arr = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

		# Gray + alpha
		elif arr.ndim == 3 and arr.shape[2] == 2:
			gray = arr[..., 0].astype(np.float32)
			alpha = arr[..., 1].astype(np.float32) / 255.0
			gray[alpha == 0] = 0.0
			gray *= alpha
			arr = np.clip(gray, 0.0, 255.0).astype(np.uint8)

		if as_gray:
			if arr.ndim == 3:
				arr = arr[..., 0]
		else:
			if arr.ndim == 2:
				arr = np.repeat(arr[..., None], 3, axis=2)

		return arr

	@staticmethod
	def _build_joint_transform():
		transforms_joint = [
			A.SquareSymmetry(p=0.5),
			A.Affine(
				p=0.75,
				scale=(0.9, 1.1),
				rotate=(-15, 15),
				interpolation=cv2.INTER_LINEAR,
				mask_interpolation=cv2.INTER_NEAREST,
			),

			# Apply the same dropout geometry to label/mask with fill_mask.
			#A.CoarseDropout(
			#	p=0.5,
			#	num_holes_range=(1, 3),
			#	fill=0,
			#	fill_mask=0,
			#	hole_height_range=(0.05, 0.2),
			#	hole_width_range=(0.05, 0.2),
			#),
		]

		return A.Compose(
			transforms_joint,
			additional_targets={
				"label": "mask",
				"valid_mask": "mask",
			},
		)

	def _apply_augmentations(self, image, label, mask):
		transformed = self._joint_transform(image=image, label=label, valid_mask=mask)
		image_aug = transformed["image"]
		label_aug = transformed["label"]
		mask_aug = transformed["valid_mask"]

		return image_aug, label_aug, mask_aug

	@staticmethod
	def _build_image_only_transform():
		# Image-only Albumentations stage. Label and mask are passed through unchanged.
		transforms_image_only = [
			
			#A.Illumination(
			#	p=1.0, 
			#	intensity_range=(0.01, 0.05)
			#	),
			A.RandomGamma(
				p=0.5, 
				gamma_limit=(90, 110)
				),
			#A.RandomBrightnessContrast(
			#	p=0.5, 
			#	brightness_limit=0.1, 
			#	contrast_limit=0.1
			#	),
		]
		return A.Compose(transforms_image_only)

	def _apply_albumentations(self, image, label, mask):
		image_aug = self._image_only_transform(image=image)["image"]
		label_aug = label
		mask_aug = mask

		return image_aug, label_aug, mask_aug

	@staticmethod
	def _add_edge_maps(image):
		r_r = roberts(image[..., 0])
		r_g = roberts(image[..., 1])
		r_b = roberts(image[..., 2])
		edges = np.stack([r_r, r_g, r_b], axis=-1).astype(image.dtype, copy=False)
		return np.concatenate([image, edges], axis=-1)
	
	@staticmethod
	def _add_edge_detection(mask, kernel_size=3):
		"""
		mask: (H, W) binary {0,1}
		returns: boundary mask (H, W)
		"""

		structure = np.ones((kernel_size, kernel_size), dtype=bool)
		eroded = binary_erosion(mask.astype(bool), structure=structure)
		boundary = np.logical_xor(mask, eroded)
		return boundary.astype(np.float32)

	@staticmethod
	def _normalize_image(image):
		return image / 255.0

	@staticmethod
	def _binarize_mask(mask, threshold):
		return (mask >= threshold).astype(np.float32)

	@staticmethod
	def _to_gray_2d(arr):
		if arr.ndim == 3:
			return arr[..., 0]
		return arr

	@staticmethod
	def _normalize_if_needed(arr):
		arr = arr.astype(np.float32, copy=False)
		if np.max(arr) > 1.0:
			arr = arr / 255.0
		return arr

	@staticmethod
	def _load_npz_sample(sample_path):
		with np.load(sample_path, allow_pickle=False) as sample:
			image = sample["image"].astype(np.float32, copy=False)
			label = sample["label"].astype(np.float32, copy=False)
			mask = sample["mask"].astype(np.float32, copy=False)
			has_label = bool(sample["has_label"]) if "has_label" in sample.files else True
			has_mask = bool(sample["has_mask"]) if "has_mask" in sample.files else True

		if image.ndim == 2:
			image = np.repeat(image[..., None], 3, axis=2)

		label = NPZSplitSegmentationDataset._to_gray_2d(label)
		mask = NPZSplitSegmentationDataset._to_gray_2d(mask)

		return image, label, mask, has_label, has_mask

	def load_instance_from_paths(
		self,
		image_path,
		label_path=None,
		mask_path=None,
	):
		if str(image_path).lower().endswith(".npz"):
			image, label, mask, has_label, has_mask = self._load_npz_sample(image_path)
		else:
			image = self._imread_zero_alpha(image_path, as_gray=False)

			if label_path is not None:
				label = self._imread_zero_alpha(label_path, as_gray=True)
				has_label = True
			else:
				label = np.full(image.shape[:2], -1, dtype=np.float32)
				has_label = False

			if mask_path is not None:
				mask = self._imread_zero_alpha(mask_path, as_gray=True)
				has_mask = True
			else:
				mask = np.ones(image.shape[:2], dtype=np.float32)
				has_mask = False
		
		# the order matters here!
		if self.apply_albumentations:
			image, label, mask = self._apply_albumentations(image, label, mask)
		
		if self.normalize_image:
			image = self._normalize_if_needed(image)
			label = self._normalize_if_needed(label)
			mask  = self._normalize_if_needed(mask)

		if self.apply_edge_maps:
			image = self._add_edge_maps(image)

		if self.apply_augmentations:
			image, label, mask = self._apply_augmentations(image, label, mask)

		if self.apply_edge_detection:
			edge_labels = self._add_edge_detection(label)  # add batch and channel dims
		else:
			edge_labels = np.zeros_like(label, dtype=np.float32)


		if self.binarize_mask:
			if has_label:
				label = self._binarize_mask(label, self.mask_threshold)
			else:
				label = np.zeros_like(label, dtype=np.float32)
			mask = self._binarize_mask(mask, self.mask_threshold)

		# cast to correct shapes: CHW for image, 1HW for label/mask
		image = np.transpose(image, (2, 0, 1))
		label = label[None, ...]
		mask = mask[None, ...]
		edge_labels = edge_labels[None, ...]  # (1,1,H,W)

		if self.use_quadrant_crop:
			image, label, mask, edge_labels = self.quad_crop(image, label, mask, edge_labels)

		# Ensure contiguous float32 views for zero-copy torch.from_numpy in __getitem__.
		image = np.ascontiguousarray(image, dtype=np.float32)
		label = np.ascontiguousarray(label, dtype=np.float32)
		mask = np.ascontiguousarray(mask, dtype=np.float32)
		edge_labels = np.ascontiguousarray(edge_labels, dtype=np.float32)

		return (
			image,
		  	label,
			mask,
			edge_labels,
			has_label, has_mask
		)


	def _move_channel_back(self, list_of_arrays):
		return [np.transpose(arr, (1, 2, 0)) for arr in list_of_arrays]
	
	def _move_channel_front(self, list_of_arrays):
		return [np.transpose(arr, (2, 0, 1)) for arr in list_of_arrays]


	#def __len__(self):
	#	return len(self.npz_files)

	def __getitem__(self, index):

		#index = random.randint(0, len(self.npz_files) - 1)
		sample_path = self.npz_files[index]
		file_name = sample_path.stem

		image, label, mask, edge_labels, has_label, has_mask = self.load_instance_from_paths(sample_path)

		return {
			"image": torch.from_numpy(image),
			"label": torch.from_numpy(label),
			"mask": torch.from_numpy(mask),
			"edge_labels": torch.from_numpy(edge_labels),
			"name": file_name,
			"has_label": torch.tensor(has_label, dtype=torch.bool),
			"has_mask": torch.tensor(has_mask, dtype=torch.bool),
		}

class RandomQuadrantCrop:
    def __init__(self, n=2):
        """
        n: grid size (N x N), must be a positive power of 2.
        n=2 is the old quadrant crop.
        """
        if not isinstance(n, int) or n <= 0 or (n & (n - 1)) != 0:
            raise ValueError("n must be a positive power of 2 (e.g., 1, 2, 4, 8, ...)")
        self.n = n

    def __call__(self, *tensors):
        """
        tensors: any number of arrays/tensors with shape (..., H, W)
        e.g. image (C,H,W), label (1,H,W), mask (1,H,W)
        """
        if len(tensors) == 0:
            raise ValueError("Expected at least one tensor")

        H = tensors[0].shape[-2]
        W = tensors[0].shape[-1]

        # Ensure all inputs share the same spatial shape.
        for t in tensors[1:]:
            if t.shape[-2] != H or t.shape[-1] != W:
                raise ValueError("All tensors must have the same H, W")

        if self.n > H or self.n > W:
            raise ValueError(f"n={self.n} is too large for input shape H={H}, W={W}")

        # Robust boundaries even when H/W are not divisible by n.
        y_edges = [int(i * H / self.n) for i in range(self.n + 1)]
        x_edges = [int(i * W / self.n) for i in range(self.n + 1)]

        row = random.randint(0, self.n - 1)
        col = random.randint(0, self.n - 1)

        y0, y1 = y_edges[row], y_edges[row + 1]
        x0, x1 = x_edges[col], x_edges[col + 1]

        slices = (..., slice(y0, y1), slice(x0, x1))
        return tuple(t[slices] for t in tensors)
	

def boundary_mask_torch(mask, kernel_size=3):
    """
    mask: (B,1,H,W) in [0,1]
    returns: boundary mask (B,1,H,W)
    """
    pad = kernel_size // 2

    # erosion via min-pooling
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)

    # XOR for soft tensors:
    boundary = mask + eroded - 2 * mask * eroded

    return boundary