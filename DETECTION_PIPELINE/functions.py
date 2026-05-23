from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image
# from cellpose import models
from cellpose_omni import models # was "from omnipose.core..."



# Cellpose segmentation parameters
MODEL_DIAMETER      = 80    # Approximate cell diameter in pixels
FLOW_THRESHOLD      = 0.6   # Boundary strictness
CELLPROB_THRESHOLD  = 0.0   # Cell detection confidence
MIN_SIZE            = 200   # Minimum cell size (filters noise)

# WBC classification thresholds
# These determine which cells are classified as WBCs vs RBCs
PURPLE_SCORE_THRESHOLD = 35     # How much more blue than green
BRIGHTNESS_THRESHOLD   = 190    # Maximum brightness for WBCs
AREA_THRESHOLD         = 1500   # Minimum area for WBCs (pixels)


def load_image(path: Path) -> np.ndarray:
    """
    Load an image as a numpy array.
    
    Supports PNG, JPG, and TIFF formats. Handles different channel orderings
    and ensures output has channels in the last dimension.
    
    Parameters:
        path: Path to the image file
        
    Returns:
        numpy array with shape (H, W) for grayscale or (H, W, C) for color
    """
    # Check file extension to determine how to load
    ext = path.suffix.lower()
    
    if ext in ['.tif', '.tiff']:
        # Load TIFF using tifffile
        img = tiff.imread(str(path))
        
        # TIFF files sometimes have channels first (C, H, W)
        # Transpose to put channels last (H, W, C)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
    
    elif ext in ['.png', '.jpg', '.jpeg']:
        # Load PNG/JPG using PIL
        pil_img = Image.open(path)
        img = np.array(pil_img)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .png, .jpg, .jpeg, .tif, or .tiff")
    
    return img



def save_image(arr: np.ndarray, path: Path):
    """
    Save a numpy array as a PNG image.
    
    Automatically handles:
    - Float arrays (rescales from 0-1 to 0-255)
    - Grayscale vs RGB images
    - Alpha channel removal
    - Directory creation
    
    Parameters:
        arr: Numpy array to save (2D grayscale or 3D RGB)
        path: Destination file path
    """
    arr_to_save = arr

    # Convert floating-point images to uint8
    if np.issubdtype(arr_to_save.dtype, np.floating):
        arr_to_save = np.clip(arr_to_save, 0, 1)
        arr_to_save = (arr_to_save * 255).astype(np.uint8)

    # Determine image mode based on array shape
    if arr_to_save.ndim == 2:
        mode = "L"  # Grayscale
    elif arr_to_save.ndim == 3 and arr_to_save.shape[2] in (3, 4):
        # Drop alpha channel if present
        if arr_to_save.shape[2] == 4:
            arr_to_save = arr_to_save[:, :, :3]
        mode = "RGB"
    else:
        raise ValueError(f"Unexpected array shape for image: {arr_to_save.shape}")

    # Create PIL image and save
    img_pil = Image.fromarray(arr_to_save, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(str(path))


print("Helper functions loaded successfully")


def extract_wbcs(img: np.ndarray,
                 masks: np.ndarray,
                 output_dir: Path,
                 image_name: str = "slide",
                 output_size: tuple = (400, 400),
                 background_color: tuple = (255, 253, 208)):  # Cream color RGB
    """
    Extract white blood cells from segmentation masks and save them.
    
    Each WBC is cropped at native resolution and centered on a 400x400 cream background.
    RBCs are filtered out and not saved.
    
    Parameters:
        img: Original image array
        masks: Cellpose segmentation mask (each cell has unique integer ID)
        output_dir: Directory to save extracted WBCs
        image_name: Base name for output files (default: "slide")
        output_size: Size of output images (width, height) - default: (400, 400)
        background_color: RGB tuple for background color - default: cream (255, 253, 208)
        
    Returns:
        Tuple of (total_cells_detected, wbcs_extracted)
    """
    # Get unique cell IDs from mask (0 = background)
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        print("No cells detected in image")
        return 0, 0
    
    wbc_count = 0
    
    # Process each detected cell
    for cell_id in unique_ids:
        # Create binary mask for this specific cell
        cell_mask = (masks == cell_id)
        area = int(cell_mask.sum())
        
        # Filter out very small detections (noise/artifacts)
        if area < MIN_SIZE:
            continue
        
        # Extract pixels belonging to this cell for color analysis
        cell_pixels = img[cell_mask]
        
        # Calculate mean RGB values
        if img.ndim == 2:
            # Grayscale: use same intensity for all channels
            mean_r = mean_g = mean_b = float(cell_pixels.mean())
        else:
            # RGB: cell_pixels shape is (N_pixels, 3)
            mean_r = float(cell_pixels[:, 0].mean())
            mean_g = float(cell_pixels[:, 1].mean())
            mean_b = float(cell_pixels[:, 2].mean())
        
        # Calculate color characteristics
        purple_score = mean_b - mean_g  # WBCs have purple/blue nuclei
        brightness = (mean_r + mean_g + mean_b) / 3.0  # Overall intensity
        
        # Classify as WBC if it meets ALL criteria
        is_wbc = (
            purple_score > PURPLE_SCORE_THRESHOLD and
            brightness < BRIGHTNESS_THRESHOLD and
            area > AREA_THRESHOLD
        )
        
        # Skip RBCs - only save WBCs
        if not is_wbc:
            continue
        
        # Get bounding box of the cell
        rows, cols = np.where(cell_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Crop the cell at native resolution (with cell mask applied)
        cell_height = max_row - min_row + 1
        cell_width = max_col - min_col + 1
        
        # Extract the cell region
        if img.ndim == 2:
            cell_crop = img[min_row:max_row+1, min_col:max_col+1].copy()
            cell_mask_crop = cell_mask[min_row:max_row+1, min_col:max_col+1]
            # Convert grayscale to RGB
            cell_crop_rgb = np.stack([cell_crop] * 3, axis=-1)
        else:
            cell_crop_rgb = img[min_row:max_row+1, min_col:max_col+1, :].copy()
            cell_mask_crop = cell_mask[min_row:max_row+1, min_col:max_col+1]
            # Handle RGBA by taking only RGB channels
            if cell_crop_rgb.shape[2] == 4:
                cell_crop_rgb = cell_crop_rgb[:, :, :3]
        
        # Ensure uint8 format
        if cell_crop_rgb.dtype != np.uint8:
            if np.issubdtype(cell_crop_rgb.dtype, np.floating):
                cell_crop_rgb = (np.clip(cell_crop_rgb, 0, 1) * 255).astype(np.uint8)
            else:
                cell_crop_rgb = cell_crop_rgb.astype(np.uint8)
        
        # Apply mask to crop (set non-cell pixels to background color)
        for c in range(3):
            cell_crop_rgb[:, :, c] = np.where(
                cell_mask_crop,
                cell_crop_rgb[:, :, c],
                background_color[c]
            )
        
        # Create output image with cream background
        output_img = np.full((output_size[1], output_size[0], 3), background_color, dtype=np.uint8)
        
        # Calculate position to center the cell
        center_y = output_size[1] // 2
        center_x = output_size[0] // 2
        
        start_y = center_y - cell_height // 2
        start_x = center_x - cell_width // 2
        
        # Ensure the cell fits within the output image
        # If cell is larger than output size, crop it from center
        if cell_height > output_size[1] or cell_width > output_size[0]:
            # Cell is too large, crop it from center
            cell_center_y = cell_height // 2
            cell_center_x = cell_width // 2
            
            crop_half_height = min(cell_center_y, output_size[1] // 2)
            crop_half_width = min(cell_center_x, output_size[0] // 2)
            
            cell_crop_y1 = cell_center_y - crop_half_height
            cell_crop_y2 = cell_center_y + crop_half_height
            cell_crop_x1 = cell_center_x - crop_half_width
            cell_crop_x2 = cell_center_x + crop_half_width
            
            cropped_cell = cell_crop_rgb[cell_crop_y1:cell_crop_y2, cell_crop_x1:cell_crop_x2]
            
            out_y1 = center_y - crop_half_height
            out_y2 = center_y + crop_half_height
            out_x1 = center_x - crop_half_width
            out_x2 = center_x + crop_half_width
            
            output_img[out_y1:out_y2, out_x1:out_x2] = cropped_cell
        else:
            # Cell fits, place it centered
            end_y = start_y + cell_height
            end_x = start_x + cell_width
            
            # Handle edge cases where cell might extend beyond boundaries
            out_start_y = max(0, start_y)
            out_start_x = max(0, start_x)
            out_end_y = min(output_size[1], end_y)
            out_end_x = min(output_size[0], end_x)
            
            cell_start_y = max(0, -start_y)
            cell_start_x = max(0, -start_x)
            cell_end_y = cell_height - max(0, end_y - output_size[1])
            cell_end_x = cell_width - max(0, end_x - output_size[0])
            
            output_img[out_start_y:out_end_y, out_start_x:out_end_x] = \
                cell_crop_rgb[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
        
        # Save the WBC
        wbc_count += 1
        out_name = f"{image_name}_WBC_{wbc_count:04d}.png"
        out_path = output_dir / out_name
        save_image(output_img, out_path)
    
    return len(unique_ids), wbc_count


print("WBC extraction function loaded successfully")



def process_slide(image_path: Path, output_dir: Path, cp_model, f: __file__):
    """
    Complete pipeline to extract WBCs from a slide image.
    
    Parameters:
        image_path: Path to input slide image (PNG, JPG, or TIFF)
        output_dir: Directory where WBCs will be saved
        
    Returns:
        Dictionary with processing statistics
    """
    print("="*60, file=f)
    print(f"Processing: {image_path.name}", file=f)
    print("="*60, file=f)
    
    # Step 1: Load the image
    print("\n[1/4] Loading image...", file=f)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = load_image(image_path)
    print(f"   Image shape: {img.shape}", file=f)
    print(f"   Image dtype: {img.dtype}", file=f)
    
    # Step 2: Run Cellpose segmentation
    print("\n[2/4] Running Cellpose segmentation...", file=f)
    masks, flows, styles = cp_model.eval(
        img,
        diameter=MODEL_DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
    )
    print("   Segmentation complete", file=f)
    
    # Step 3: Extract WBCs
    print("\n[3/4] Extracting and classifying cells...", file=f)
    total_cells, wbc_count = extract_wbcs(
        img,
        masks,
        output_dir,
        image_name=image_path.stem
    )
    
    # Step 4: Summary
    print("\n[4/4] Processing complete!", file=f)
    print("\n" + "="*60, file=f)
    print("SUMMARY", file=f)
    print("="*60, file=f)
    print(f"Total cells detected:  {total_cells}", file=f)
    print(f"WBCs extracted:        {wbc_count}", file=f)
    print(f"RBCs filtered out:     {total_cells - wbc_count}", file=f)
    print(f"\nOutput directory:      {output_dir}", file=f)
    print("="*60, file=f)
    
    return {
        'total_cells': total_cells,
        'wbcs_extracted': wbc_count,
        'rbcs_filtered': total_cells - wbc_count,
        'output_dir': str(output_dir)
    }


print("Main pipeline function loaded successfully")
