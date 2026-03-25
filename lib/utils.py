# Display the first 25 matched images and labels in a 5x5 grid with minimal margins
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from skimage.morphology import opening, closing, footprint_rectangle
from scipy.ndimage import find_objects, label as ndi_label, sobel


"""
def show_image_label_pairs(dir='data',
                           image_folder='images',
                           label_folder='labels'
                           ):
    # Define paths
    images_dir = os.path.join(dir, image_folder)
    labels_dir = os.path.join(dir, label_folder)

    # Get sorted list of image and label files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])

    # Match images and labels by filename
    matched = [(img, img) for img in image_files if img in label_files]

    # Select first 25 pairs
    matched = matched[:25]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(5, 10, wspace=0.02, hspace=0.02)

    for idx, (img_name, lbl_name) in enumerate(matched):
        row, col = divmod(idx, 5)
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, lbl_name)
        image = imageio.imread(img_path)
        label = imageio.imread(lbl_path)

        ax_img = fig.add_subplot(gs[row, col])
        ax_lbl = fig.add_subplot(gs[row, col + 5])

        ax_img.imshow(image, 
                      interpolation='none')
        ax_img.axis('off')

        ax_lbl.imshow(label, 
                      interpolation='none',
                      cmap='gray', 
                      vmin=0, vmax=255,
                      )
        ax_lbl.axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()
"""

# DATA CLEANING ======================================================================

def clean_dataset(input_dir='data/original',
                  output_dir='data/cleaned',
                  image_folder='images',
                  label_folder='labels',
                  masks_folder='masks',
                  files_to_drop=None,
                  
                  threshold_value=0.5,
                  morph_size=3,
                  ):
    
    images_dir = os.path.join(input_dir, image_folder)
    labels_dir = os.path.join(input_dir, label_folder)
    masks_dir = os.path.join(input_dir, masks_folder)

    images_cleaned_dir = os.path.join(output_dir, image_folder)
    labels_cleaned_dir = os.path.join(output_dir, label_folder)
    masks_cleaned_dir = os.path.join(output_dir, masks_folder)

    for out_dir in [images_cleaned_dir, labels_cleaned_dir, masks_cleaned_dir]:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=False)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])

    # drop files to skip based on filename (without extension) before the .png part
    # we also need to strip the part before the image name if the files are in subfolders, so we only compare the base filename
    if files_to_drop is not None:
        image_files = [f for f in image_files if os.path.splitext(f)[0] not in files_to_drop]
        label_files = [f for f in label_files if os.path.splitext(f)[0] not in files_to_drop]

    # perform cleaning for images
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)

        # read the image
        image = imageio.imread(img_path)

        # correct the alpha channel if present
        image, mask = alpha_conversion_masking(image)
        mask = morphological_cleanup(mask)
        mask = mask.astype(np.uint8) * 255

        # Copy image to cleaned folder (or apply any desired transformations here)
        imageio.imwrite(os.path.join(images_cleaned_dir, img_name), image)
        imageio.imwrite(os.path.join(masks_cleaned_dir, img_name), mask)

    # perform cleaning for labels
    for lbl_name in label_files:
        lbl_path = os.path.join(labels_dir, lbl_name)

        # read the label
        label = imageio.imread(lbl_path)

        # correct the alpha channel if present
        label, _ = alpha_conversion_masking(label, as_gray=True, normalize=True)

        # 1) Binarize on normalized [0, 1] labels
        binary_label = (label >= threshold_value)

        # 2) Tiny morphology cleanup: opening then closing
        binary_label = morphological_cleanup(binary_label, morph_size=morph_size)

        # Save cleaned binary label as uint8 PNG (0 or 255)
        out_label = binary_label.astype(np.uint8) * 255
        imageio.imwrite(os.path.join(labels_cleaned_dir, lbl_name), out_label)


def alpha_conversion_masking(arr, as_gray=False, normalize=False):
    """
    Process image with alpha channel and return:
    - image (RGB or grayscale)
    - valid_mask (1 = valid, 0 = ignore)
    """

    valid_mask = None

    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3].astype(np.float32) / 255.0

        # Valid pixels: alpha > 0
        valid_mask = (alpha > 0)

        # Premultiply alpha
        rgb *= alpha[..., None]

        arr = np.clip(rgb, 0.0, 255.0)

        if as_gray:
            # Proper luminance conversion
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            arr = 0.299 * r + 0.587 * g + 0.114 * b

    elif arr.ndim == 3 and arr.shape[-1] == 2:
        gray = arr[..., 0].astype(np.float32)
        alpha = arr[..., 1].astype(np.float32) / 255.0

        valid_mask = (alpha > 0)

        gray *= alpha
        arr = np.clip(gray, 0.0, 255.0)

    else:
        # No alpha channel → everything valid
        valid_mask = np.ones(arr.shape[:2], dtype=np.float32)

        if as_gray and arr.ndim == 3:
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            arr = 0.299 * r + 0.587 * g + 0.114 * b

    # Normalize if requested
    if normalize:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.uint8)

    return arr, valid_mask.astype(np.uint8)


def morphological_cleanup(binary_mask, morph_size=3):
    """
    Apply morphological opening followed by closing to clean up a binary mask.

    Args:
        binary_mask (np.ndarray): Input binary mask (boolean or 0/1).
        morph_size (int): Size of the structuring element for morphology.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    footprint = footprint_rectangle((morph_size, morph_size))
    cleaned_mask = opening(binary_mask, footprint=footprint)
    cleaned_mask = closing(cleaned_mask, footprint=footprint)
    return cleaned_mask


def compute_cleaned_mask_sizes(dir='data',
                               cleaned_label_folder='labels'):
    """
    Compute connected-object sizes for each cleaned binary mask.

    Returns:
        list[list[int]]: per-image list of connected object pixel counts.
    """
    cleaned_dir = os.path.join(dir, cleaned_label_folder)

    if not os.path.isdir(cleaned_dir):
        raise FileNotFoundError(f"Folder not found: {cleaned_dir}")

    label_files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith('.png')])
    if len(label_files) == 0:
        return []

    all_counts = []
    for lbl_name in label_files:
        lbl_path = os.path.join(cleaned_dir, lbl_name)
        binary_label = imageio.imread(lbl_path, mode='L')  # Load as grayscale
        #binary_label = (label > 0)

        labeled_mask, _ = ndi_label(binary_label)
        slices = find_objects(labeled_mask)

        object_counts = []
        for comp_id, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            component = (labeled_mask[slc] == comp_id)
            object_counts.append(int(np.count_nonzero(component)))

        all_counts.append(object_counts)

    return all_counts


# CORRELATION ANALYSIS ===============================================================

def show_sobel_mask_alignment(dir='data',
                              image_folder='images',
                              label_folder='labels_cleaned',
                              index=0):
    """
    Show Sobel edge magnitude and overlay binary-mask contours to inspect alignment.
    """
    images_dir = os.path.join(dir, image_folder)
    labels_dir = os.path.join(dir, label_folder)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    matched = [(img, img) for img in image_files if img in label_files]
    matched = matched[:25]

    if len(matched) == 0:
        raise ValueError("No matched image-label pairs found.")
    if index < 0 or index >= len(matched):
        raise IndexError(f"index must be in [0, {len(matched)-1}], got {index}")

    img_name, lbl_name = matched[index]
    img_path = os.path.join(images_dir, img_name)
    lbl_path = os.path.join(labels_dir, lbl_name)

    image = imread_zero_alpha(img_path, as_gray=False, normalize=False)
    label = imread_zero_alpha(lbl_path, as_gray=True, normalize=False)
    binary_label = (label > 0).astype(np.uint8)

    # Convert RGB image to grayscale and compute Sobel magnitude.
    if image.ndim == 3:
        gray = image.astype(np.float32).mean(axis=2)
    else:
        gray = image.astype(np.float32)

    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    max_val = np.max(sobel_mag)
    if max_val > 0:
        sobel_mag = sobel_mag / max_val

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image, interpolation='none')
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(sobel_mag, cmap='gray', interpolation='none')
    axes[1].contour(binary_label, levels=[0.5], colors='red', linewidths=0.8, alpha=0.5)
    axes[1].set_title('Sobel + Mask Contour')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def compute_masked_pixel_colors(dir='data/cleaned',
                                image_folder='images',
                                mask_folder='masks',
                                label_folder='labels',
                                ):
    """
    Collect RGB colors of pixels where the corresponding cleaned mask is foreground.

    Returns:
        list[np.ndarray]: one array per matched image, each with shape (N_i, 3).
    """
    image_dir = os.path.join(dir, image_folder)
    masks_dir = os.path.join(dir, mask_folder)
    label_dir = os.path.join(dir, label_folder)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Folder not found: {image_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Folder not found: {masks_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Folder not found: {label_dir}")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = set([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    label_files = set([f for f in os.listdir(label_dir) if f.endswith('.png')])

    matched = [f for f in image_files if f in mask_files and f in label_files]
    if len(matched) == 0:
        return []

    all_colors = []
    all_labels = []
    for file_name in matched:
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(masks_dir, file_name)
        label_path = os.path.join(label_dir, file_name)

        image = imageio.imread(image_path)
        mask  = imageio.imread(mask_path)
        label = imageio.imread(label_path)

        image = image / 255
        mask  = mask  / 255
        label = label / 255

        # retrieve colors both for 0 and 1 label pixels, but only where mask is valid
        mask = mask.astype(np.bool)
        colors = image[mask]
        labels = label[mask]

        all_colors.append(colors)
        all_labels.append(labels)        

    all_colors = np.concatenate(all_colors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_colors, all_labels



def _ensure_float(X):
    return X.astype(np.float32)

def _safe_divide(a, b, eps=1e-8):
    return a / (b + eps)

def luminance_rec601(X):
    """Y = 0.299R + 0.587G + 0.114B"""
    X = _ensure_float(X)
    return 0.299*X[:, 0] + 0.587*X[:, 1] + 0.114*X[:, 2]

def luminance_rec709(X):
    """Y = 0.2126R + 0.7152G + 0.0722B"""
    X = _ensure_float(X)
    return 0.2126*X[:, 0] + 0.7152*X[:, 1] + 0.0722*X[:, 2]

def chromaticity(X):
    """
    Returns (r, g) where:
    r = R / (R+G+B)
    g = G / (R+G+B)
    """
    X = _ensure_float(X)
    s = X.sum(axis=1)
    
    r = _safe_divide(X[:, 0], s)
    g = _safe_divide(X[:, 1], s)
    
    return np.stack([r, g], axis=1)

def excess_green(X):
    """ExG = 2G - R - B"""
    X = _ensure_float(X)
    return 2*X[:, 1] - X[:, 0] - X[:, 2]

def intensity_color_deviation(X):
    """
    Returns (Y, R-Y, G-Y)
    """
    X = _ensure_float(X)
    Y = luminance_rec709(X)
    
    r_dev = X[:, 0] - Y
    g_dev = X[:, 1] - Y
    
    return np.stack([Y, r_dev, g_dev], axis=1)

def rgb_to_hsv(X):
    """
    Returns (H, S, V)
    H in [0,1], S in [0,1], V in [0,1]
    """
    X = _ensure_float(X)

    # normalize if needed
    if X.max() > 1.0:
        X = X / 255.0

    r, g, b = X[:, 0], X[:, 1], X[:, 2]

    maxc = np.max(X, axis=1)
    minc = np.min(X, axis=1)
    delta = maxc - minc

    # Hue
    H = np.zeros_like(maxc)

    mask = delta > 1e-8

    # different cases
    idx = (maxc == r) & mask
    H[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6

    idx = (maxc == g) & mask
    H[idx] = (b[idx] - r[idx]) / delta[idx] + 2

    idx = (maxc == b) & mask
    H[idx] = (r[idx] - g[idx]) / delta[idx] + 4

    H = H / 6.0  # normalize to [0,1]

    # Saturation
    S = _safe_divide(delta, maxc)

    # Value
    V = maxc

    return np.stack([H, S, V], axis=1)


def features_luminance(X):
    return luminance_rec709(X)[:, None]

def features_hsv(X):
    return rgb_to_hsv(X)

def features_chromaticity(X):
    return chromaticity(X)

def features_exg_luminance(X):
    Y = luminance_rec709(X)
    exg = excess_green(X)
    return np.stack([exg, Y], axis=1)

def features_rgb_luminance(X):
    Y = luminance_rec709(X)
    return np.concatenate([X, Y[:, None]], axis=1)








def load_images_and_masks(dir='data',
                          image_folder='images',
                          cleaned_label_folder='labels_cleaned',
                          normalize_images=True):
    """
    Load matched images and cleaned binary masks.

    Returns:
        images: list of arrays, each shape (H, W, 3)
        masks: list of arrays, each shape (H, W), values in {0, 1}
    """
    images_dir = os.path.join(dir, image_folder)
    labels_dir = os.path.join(dir, cleaned_label_folder)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Folder not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Folder not found: {labels_dir}")

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    label_files = set([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    matched = [f for f in image_files if f in label_files]

    if len(matched) == 0:
        return [], []

    images = []
    masks = []
    for file_name in matched:
        img_path = os.path.join(images_dir, file_name)
        lbl_path = os.path.join(labels_dir, file_name)

        img = imread_zero_alpha(img_path, as_gray=False, normalize=False)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[..., :3]

        mask = imread_zero_alpha(lbl_path, as_gray=True, normalize=False)
        mask = (mask > 0).astype(np.uint8)

        if normalize_images:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        images.append(img)
        masks.append(mask)

    return images, masks


def extract_pixels_from_lists(images, masks, max_samples=100_000, seed=None):
    """
    Convert lists of images/masks into pixel-level feature matrix X and labels y.

    Args:
        images: list of arrays, shape (H, W, 3)
        masks: list of arrays, shape (H, W), values {0, 1}
        max_samples: optional cap for random subsampling
        seed: optional random seed for deterministic subsampling

    Returns:
        X: np.ndarray shape (N, 3)
        y: np.ndarray shape (N,)
    """
    if len(images) != len(masks):
        raise ValueError("images and masks must have the same length")

    X_chunks = []
    y_chunks = []

    for img, mask in zip(images, masks):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Each image must have shape (H, W, 3)")
        if mask.ndim != 2:
            raise ValueError("Each mask must have shape (H, W)")
        if img.shape[:2] != mask.shape:
            raise ValueError("Image and mask spatial dimensions must match")

        pixels = img.reshape(-1, 3)
        labels = mask.reshape(-1)

        X_chunks.append(pixels)
        y_chunks.append(labels)

    if not X_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    X = np.vstack(X_chunks).astype(np.float32)
    y = np.concatenate(y_chunks).astype(np.uint8)

    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


def fft_pool_image(image, factor=2, mode="mask"):
    """
    Apply frequency pooling using a 2D Hartley transform.

    mode:
        - "mask": old behavior (zero-masked full-size spectrum)
        - "crop": no zero insertion, inverse-transform cropped spectrum
    """
    if factor < 1:
        return image

    x = np.asarray(image)
    if x.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    if mode not in ("mask", "crop"):
        raise ValueError("mode must be 'mask' or 'crop'")

    orig_dtype = x.dtype
    x = x.astype(np.float32, copy=False)

    def _hartley_2d(arr2d):
        spec = np.fft.fft2(arr2d)
        return (spec.real - spec.imag).astype(np.float32, copy=False)

    def _ihartley_2d(arr2d):
        h, w = arr2d.shape
        return _hartley_2d(arr2d) / float(h * w)

    def _hartley_pool_2d(channel):
        h, w = channel.shape

        freq = _hartley_2d(channel)
        freq = np.fft.fftshift(freq)

        keep_h = max(1, h // factor)
        keep_w = max(1, w // factor)

        cy, cx = h // 2, w // 2
        y0 = cy - keep_h // 2
        x0 = cx - keep_w // 2
        y1 = y0 + keep_h
        x1 = x0 + keep_w

        if mode == "mask":
            mask = np.zeros((h, w), dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            filtered = freq * mask
            filtered = np.fft.ifftshift(filtered)
            pooled = _ihartley_2d(filtered)
        else:
            # No zero insertion: inverse-transform cropped spectrum directly
            cropped = freq[y0:y1, x0:x1]
            cropped = np.fft.ifftshift(cropped)
            pooled = _ihartley_2d(cropped)

        return pooled

    if x.ndim == 2:
        out = _hartley_pool_2d(x)
    else:
        channels = [_hartley_pool_2d(x[..., c]) for c in range(x.shape[2])]
        out = np.stack(channels, axis=-1)

    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(out, 0, 255).astype(orig_dtype)
    else:
        out = out.astype(orig_dtype, copy=False)

    return out
