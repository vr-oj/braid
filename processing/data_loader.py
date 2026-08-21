import os
import cv2
import json
import logging
import getpass
import numpy as np
from tifffile import TiffFile
from skimage import img_as_float
from skimage.segmentation import chan_vese
from skimage.draw import rectangle, ellipse
from PySide6.QtCore import QRect
from processing.resource_loader import resource_path
from processing.trimming import iter_selected_frames
import concurrent.futures
import multiprocessing
from pathlib import Path
import csv

log = logging.getLogger(__name__)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

BUTI_CSV_SETTINGS = {
    "experiment_type": ("experiment_type", str),
    "preload_mm": ("preload_mm", float),
    "deformation_mm": ("deformation_mm", float),
    "deformation_percent": ("deformation_percent", float),
    "steps": ("steps", lambda value: int(float(value))),
    "rate_forward_mm_s": ("rate_forward_mm_s", float),
    "rate_reverse_mm_s": ("rate_reverse_mm_s", float),
    "cycles_configured": ("cycles", lambda value: int(float(value))),
    "wire_diameter_mm": ("wire_diameter_mm", float),
    "constant_tension_mn": ("constant_tension_mn", float),
}


def get_system_username():
    """Returns the current system user name in a cross-platform way."""
    try:
        return getpass.getuser()
    except Exception:
        # Fallbacks: Windows, Unix, etc.
        return os.environ.get('USERNAME') or os.environ.get('USER') or os.environ.get('LOGNAME') or None


def load_colors():
    path = resource_path("resources/colors.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frame_loader(signals, file_path, frame_indices, count=False):
    """
    Loads specific frames from a video or multi-page TIFF file, handling errors gracefully.

    This function checks the file extension to determine the loading method. For TIFF files,
    it uses the tifffile library to directly access frames by index. For all other file
    types, it uses OpenCV's VideoCapture.
    """
    loaded_frames = {}
    file_ext = os.path.splitext(file_path)[1].lower()

    # --- TIFF File Handling ---
    if file_ext in ['.tif', '.tiff']:
        try:
            with TiffFile(file_path) as tif:
                frame_count = len(tif.pages)
                signals.message.emit("Collecting TIFF image data...")
                log.info(f"Starting frame extraction for {len(frame_indices)} frames from {file_path}: {frame_indices}")

                for i, f in enumerate(frame_indices):
                    try:
                        if f >= frame_count:
                            log.warning(
                                f"Frame index {f} is out of bounds for TIFF with {frame_count} pages. Skipping.")
                            continue

                        frame = tif.pages[f].asarray()

                        # Ensure the frame is grayscale for consistent processing
                        if frame.ndim == 3:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Handles RGB or multi-channel images
                        else:
                            gray = frame.copy()  # It's already grayscale, create a copy

                        # Normalize to 8-bit uint for compatibility with downstream processes
                        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
                        gray = gray.astype(np.uint8)

                        # Embed frame count into the first row of pixels
                        if len(str(frame_count)) < gray.shape[1]:
                            for j, digit in enumerate(str(frame_count)):
                                gray[0, j] = int(digit)
                        else:
                            log.warning(f"Frame {f}: Not enough horizontal pixels to write metadata.")

                        loaded_frames[f] = gray

                        pct = int(((i + 1) / len(frame_indices)) * 100)
                        signals.progress.emit(pct)

                    except Exception as e:
                        # Log error for a single frame and continue with the next
                        log.error(f"Error processing TIFF frame at index {f}: {e}", exc_info=True)
                        signals.message.emit(f"Error on TIFF frame {f}, see log for details.")

                if count:
                    loaded_frames[frame_count] = None

                    embedded_data = {}

                    for i, pg in enumerate(tif.pages):
                        try:
                            desc = pg.tags.get("ImageDescription")
                            if not desc:
                                log.debug(f"No deviceTime in first page, stopping search.")
                                continue
                            try:
                                info = json.loads(desc.value)
                            except Exception:
                                continue
                            keys = list(info.keys())

                            for k in keys:
                                if k not in embedded_data:
                                    embedded_data[k] = []
                                embedded_data[k].append(info[k])

                            # "time_s"
                            # "frameIdx"
                            # "distance"
                            # "cycle"
                            # "force"

                            pct = int(((i + 1) / frame_count) * 100)
                            signals.progress.emit(pct)

                        except Exception as e:
                            # Log error for a single frame and continue with the next
                            log.error(f"Error processing TIFF frame at index {i}: {e}", exc_info=True)
                            signals.message.emit(f"Error on TIFF frame {i}, see log for details.")

                    loaded_frames["data"] = embedded_data

        except Exception as e:
            err_msg = f"Failed to open or process TIFF file: {file_path}. Error: {e}"
            log.error(err_msg, exc_info=True)
            signals.message.emit(err_msg)
            raise IOError(err_msg)

    # --- Video File Handling (Original Logic) ---

    elif file_ext in [".mkv"]:

        vid = cv2.VideoCapture(file_path)
        if not vid.isOpened():
            err_msg = f"Failed to open video file: {file_path}"
            log.error(err_msg)
            signals.message.emit(err_msg)
            raise IOError(err_msg)

        try:
            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            signals.message.emit("Collecting frame data...")
            log.info(f"Starting frame extraction for {len(frame_indices)} frames from {file_path}")

            for i, f in enumerate(frame_indices):
                try:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
                    res, frame = vid.read()

                    if not res:
                        log.warning(f"Could not read frame at index {f}. Skipping.")
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)

                    # Embed frame count using the same logic as for TIFFs
                    if len(str(frame_count)) < gray.shape[1]:
                        for j, digit in enumerate(str(frame_count)):
                            gray[0, j] = int(digit)
                    else:
                        log.warning(f"Frame {f}: Not enough horizontal pixels to write metadata.")

                    loaded_frames[f] = gray

                    pct = int(((i + 1) / len(frame_indices)) * 100)
                    signals.progress.emit(pct)

                except Exception as e:
                    log.error(f"Error processing frame at index {f}: {e}", exc_info=True)
                    signals.message.emit(f"Error on frame {f}, see log for details.")

            if count:
                loaded_frames[frame_count] = None
        finally:
            # Ensure the video capture is always released
            log.info(f"Finished frame extraction. Releasing video capture for {file_path}.")
            vid.release()


    else:
        raise "Other video files not yet supported. Contact customer support for additional help."

    signals.progress.emit(100)
    signals.message.emit("Frame processing complete.")
    return loaded_frames


def parse_and_validate_csv(csv_path: Path) -> dict | None:
    """Read required telemetry and optional BURST/BUTI acquisition settings."""
    required_cols = {"time_s", "frame_index", "distance", "cycle", "force"}

    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])

            if not required_cols.issubset(headers):
                log.warning(f"CSV missing columns. Expected {required_cols}, found {headers}")
                return None

            # Keep the established numeric telemetry shape. Acquisition
            # settings are normalized separately so analysis code cannot
            # accidentally treat them as per-sample numeric signals.
            data = {col: [] for col in required_cols}
            buti_settings = {}

            # Populate the dictionary
            for row in reader:
                for col in required_cols:
                    # Convert to float (or int for frame_index if preferred)
                    data[col].append(float(row[col]))

                for csv_column, (settings_key, converter) in BUTI_CSV_SETTINGS.items():
                    value = (row.get(csv_column) or "").strip()
                    if not value or settings_key in buti_settings:
                        continue
                    try:
                        buti_settings[settings_key] = converter(value)
                    except (TypeError, ValueError):
                        log.warning(
                            "Ignoring invalid BUTI setting %s=%r in %s",
                            csv_column,
                            value,
                            csv_path,
                        )

            if buti_settings:
                data["buti_settings"] = buti_settings

            return data

    except Exception as e:
        log.error(f"Failed to parse CSV {csv_path}: {e}")
        return None


def extract_buti_settings(data: dict | None) -> dict:
    """Normalize settings loaded from a BURST CSV or embedded TIFF metadata."""

    if not isinstance(data, dict):
        return {}

    raw_settings = data.get("buti_settings")
    if isinstance(raw_settings, dict):
        return dict(raw_settings)

    # TIFF metadata is collected once per page, so a settings snapshot arrives
    # as a repeated list. All pages in one run carry the same snapshot.
    if isinstance(raw_settings, list):
        for item in raw_settings:
            if isinstance(item, dict) and item:
                return dict(item)

    return {}


def _interpolate_rois_worker(roi_data: dict, pct: float) -> list[dict]:
    """Pure function to interpolate ROIs safely in a background thread."""

    def lerp_int(start, end, pct):
        return int(round(start + (end - start) * pct))

    def lerp_rect(r_min, r_max, pct):
        # Unpack the standard tuples
        x1, y1, w1, h1 = r_min
        x2, y2, w2, h2 = r_max

        # Return a new interpolated tuple instead of a QRect
        return (
            lerp_int(x1, x2, pct),
            lerp_int(y1, y2, pct),
            lerp_int(w1, w2, pct),
            lerp_int(h1, h2, pct)
        )

    interpolated_rois = []
    for i in range(2):
        min_roi = roi_data["min"][i]
        max_roi = roi_data["max"][i]
        interp_rect = lerp_rect(min_roi["roi_rect"], max_roi["roi_rect"], pct)

        shape_type = min_roi["seed_shape_type"]
        c_min = min_roi["seed_coords"]
        c_max = max_roi["seed_coords"]

        interp_coords = {}
        if shape_type == 'rect':
            interp_coords = {
                'x': lerp_int(c_min['x'], c_max['x'], pct),
                'y': lerp_int(c_min['y'], c_max['y'], pct),
                'w': lerp_int(c_min['w'], c_max['w'], pct),
                'h': lerp_int(c_min['h'], c_max['h'], pct)
            }
        elif shape_type == 'ellipse':
            interp_coords = {
                'center_x': lerp_int(c_min['center_x'], c_max['center_x'], pct),
                'center_y': lerp_int(c_min['center_y'], c_max['center_y'], pct),
                'radius_x': lerp_int(c_min['radius_x'], c_max['radius_x'], pct),
                'radius_y': lerp_int(c_min['radius_y'], c_max['radius_y'], pct)
            }

        interpolated_rois.append({
            "roi_rect": interp_rect,
            "seed_shape_type": shape_type,
            "seed_coords": interp_coords
        })
    return interpolated_rois


# def geometry_worker(signals, config: dict):
#     """
#     The heavy-lifting background task.
#     Reads the TIFF file, applies Chan-Vese, and calculates dimensional math.
#     """
#     file_path = config['file_path']
#     distances = config['distances']
#     min_dist = config['min_dist']
#     max_dist = config['max_dist']
#     roi_data = config['roi_data']
#
#     mu = config['mu']
#     gamma = config['gamma']
#     lambda1 = config['lambda1']
#
#     frames_out = []
#     width_masks_out, thickness_masks_out = [], []
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#
#     try:
#         with TiffFile(file_path) as tif:
#             total_frames = len(distances)
#             signals.message.emit("Calculating Geometry over all frames...")
#
#             for i, distance in enumerate(distances):
#                 # 1. Calculate percentage and clamp it mathematically
#                 pct = (distance - min_dist) / (max_dist - min_dist)
#                 pct = np.clip(pct, 0.0, 1.0)
#
#                 # 2. Get interpolated boxes and seeds
#                 interp_rois = _interpolate_rois_worker(roi_data, pct)
#
#                 # 3. Extract Image Data
#                 frame = tif.pages[i].asarray()
#                 if frame.ndim == 3:
#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 else:
#                     gray = frame.copy()
#
#                 # if i % 10 == 0:  # Only log every 10th frame so we don't spam the console  # Debugging
#                 #     log.info(f"--- FRAME {i} | Type {gray.dtype} ---")
#                 #     log.info(f"1. Frame min: {gray.min()}")
#                 #     log.info(f"1. Frame max: {gray.max()}")
#
#                 cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
#                 gray = gray.astype(np.uint8)
#
#                 dimensions = []  # Will hold [width_px, length_px]
#
#                 # 4. Process both ROIs (0 = Width, 1 = Length)
#                 for roi_idx, roi in enumerate(interp_rois):
#                     # Crop
#                     r = roi['roi_rect']
#                     crop = gray[r.y(): r.y() + r.height(), r.x(): r.x() + r.width()]
#
#                     if crop.size == 0:
#                         dimensions.append(0)
#                         # Ensure we append empty masks to maintain index parity
#                         empty_mask = {'mask': np.array([], dtype=bool), 'offset_x': 0, 'offset_y': 0}
#                         if roi_idx == 0:
#                             width_masks_out.append(empty_mask)
#                         else:
#                             thickness_masks_out.append(empty_mask)
#                         continue
#
#                     # Generate Seed Mask
#                     mask_shape = crop.shape
#                     seed_mask = np.zeros(mask_shape, dtype=bool)
#                     coords = roi['seed_coords']
#
#                     if roi['seed_shape_type'] == 'rect':
#                         start = (coords['y'], coords['x'])
#                         end = (coords['y'] + coords['h'], coords['x'] + coords['w'])
#                         rr, cc = rectangle(start=start, end=end, shape=mask_shape)
#                         seed_mask[rr, cc] = True
#                     elif roi['seed_shape_type'] == 'ellipse':
#                         rr, cc = ellipse(coords['center_y'], coords['center_x'],
#                                          coords['radius_y'], coords['radius_x'], shape=mask_shape)
#                         seed_mask[rr, cc] = True
#
#                     # Run Chan-Vese
#                     img_float = img_as_float(crop)
#                     if gamma != 1.0:
#                         img_float = img_float ** gamma
#
#                     cv_result = chan_vese(img_float, mu=mu, lambda1=lambda1, lambda2=1.0,
#                                           tol=1e-3, max_num_iter=100, dt=0.5,
#                                           init_level_set=seed_mask, extended_output=True)  # Ensure tuple output
#
#                     # Convert to binary mask (0 or 1)
#                     final_mask = cv_result[0].astype(np.uint8)
#
#                     # Optional Morphology cleanup (ensure mask is 0/255 for cv2)
#                     cleaned_mask = cv2.morphologyEx(final_mask * 255, cv2.MORPH_OPEN, kernel)
#                     binary_mask = cleaned_mask // 255
#
#                     # if i % 10 == 0:  # Only log every 10th frame so we don't spam the console  ## Debugging
#                     #     log.info(f"--- FRAME {i} | ROI {roi_idx} ---")
#                     #     log.info(f"1. Crop Shape: {crop.shape}")
#                     #     log.info(f"2. Seed Pixels: {np.sum(seed_mask)}")
#                     #     log.info(f"3. Chan-Vese Pixels: {np.sum(final_mask)}")
#                     #     log.info(f"4. Cleaned Pixels: {np.sum(binary_mask)}")
#
#                     # --- MASK CROPPING & STORAGE ---
#                     # Find coordinates of all non-zero pixels
#                     y_idx, x_idx = np.nonzero(binary_mask)
#
#                     if len(y_idx) > 0:
#                         # Get bounding box of the active mask
#                         min_y, max_y = y_idx.min(), y_idx.max()
#                         min_x, max_x = x_idx.min(), x_idx.max()
#
#                         # Slice the array and cast to boolean (1 byte per pixel)
#                         tight_mask = binary_mask[min_y:max_y + 1, min_x:max_x + 1].astype(bool)
#                         mask_data = {
#                             'mask': tight_mask,
#                             'offset_x': int(min_x),
#                             'offset_y': int(min_y)
#                         }
#                     else:
#                         mask_data = {
#                             'mask': np.array([], dtype=bool),
#                             'offset_x': 0,
#                             'offset_y': 0
#                         }
#                         # -------------------------------
#
#                     # Simply store the masks based on the ROI index
#                     if roi_idx == 0:
#                         width_masks_out.append(mask_data)
#                     else:
#                         thickness_masks_out.append(mask_data)
#
#                     # Both ROIs for this frame are processed. Log the frame index.
#                 frames_out.append(i)
#
#                 # Update Progress
#                 progress_pct = int(((i + 1) / total_frames) * 100)
#                 signals.progress.emit(progress_pct)
#
#             signals.message.emit("Geometry calculation complete.")
#
#             # Return only the frames and the raw segmentation masks
#             return {
#                 #'idx': frames_out,
#                 'first_masks': width_masks_out,
#                 'second_masks': thickness_masks_out
#             }
#
#     except Exception as e:
#         signals.message.emit(f"Error calculating geometry: {str(e)}")
#         raise e
#
# def geometry_worker(signals, config: dict):
#     file_path = config['file_path']
#     distances = config['distances']
#     min_dist = config['min_dist']
#     max_dist = config['max_dist']
#     roi_data = config['roi_data']
#
#     mu = config['mu']
#     gamma = config['gamma']
#     lambda1 = config['lambda1']
#
#     frames_out = []
#     width_masks_out, thickness_masks_out = [], []
#
#     # Get optimal core count (leave one or two for the OS/GUI)
#     max_workers = max(1, multiprocessing.cpu_count() - 2)
#
#     # We will store results in dictionaries since futures complete out of order
#     width_results = {}
#     thickness_results = {}
#
#     with TiffFile(file_path) as tif:
#         total_frames = len(distances)
#
#         # Start the multiprocessing pool
#         with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = []
#
#             signals.message.emit("Reading image data and queuing tasks...")
#
#             # --- PRODUCER LOOP ---
#             for i, distance in enumerate(distances):
#                 # 1. Calculate percentage and clamp it mathematically
#                 pct = (distance - min_dist) / (max_dist - min_dist)
#                 pct = np.clip(pct, 0.0, 1.0)
#
#                 # 2. Get interpolated boxes and seeds
#                 interp_rois = _interpolate_rois_worker(roi_data, pct)
#
#                 frame = tif.pages[i].asarray()
#                 if frame.ndim == 3:
#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 else:
#                     gray = frame.copy()
#
#                 cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
#                 gray = gray.astype(np.uint8)
#                 dimensions = []  # Will hold [width_px, length_px]
#
#                 for roi_idx, roi in enumerate(interp_rois):
#                     # 1. CROP THE IMAGE HERE (In the Manager)
#                     r = roi['roi_rect']
#                     crop = gray[r.y(): r.y() + r.height(), r.x(): r.x() + r.width()]
#
#                     # 2. GENERATE SEED MASK HERE
#                     if crop.size == 0:
#                         dimensions.append(0)
#                         # Ensure we append empty masks to maintain index parity
#                         empty_mask = {'mask': np.array([], dtype=bool), 'offset_x': 0, 'offset_y': 0}
#                         if roi_idx == 0:
#                             width_masks_out.append(empty_mask)
#                         else:
#                             thickness_masks_out.append(empty_mask)
#                         continue
#
#                     # Generate Seed Mask
#                     mask_shape = crop.shape
#                     seed_mask = np.zeros(mask_shape, dtype=bool)
#                     coords = roi['seed_coords']
#
#                     if roi['seed_shape_type'] == 'rect':
#                         start = (coords['y'], coords['x'])
#                         end = (coords['y'] + coords['h'], coords['x'] + coords['w'])
#                         rr, cc = rectangle(start=start, end=end, shape=mask_shape)
#                         seed_mask[rr, cc] = True
#                     elif roi['seed_shape_type'] == 'ellipse':
#                         rr, cc = ellipse(coords['center_y'], coords['center_x'],
#                                          coords['radius_y'], coords['radius_x'], shape=mask_shape)
#                         seed_mask[rr, cc] = True
#
#                     # 3. PACKAGE THE TINY PAYLOAD
#                     payload = {
#                         'crop': crop,
#                         'seed_mask': seed_mask,
#                         'mu': mu, 'gamma': gamma, 'lambda1': lambda1,
#                         'frame_idx': i,
#                         'roi_idx': roi_idx
#                     }
#
#                     # 4. SUBMIT TO WORKER POOL
#                     future = executor.submit(compute_chan_vese_worker, payload)
#                     futures.append(future)
#
#                     progress_pct = int(((i + 1) / total_frames) * 50)
#                     signals.progress.emit(progress_pct)
#
#             signals.message.emit("Processing segmentation masks...")
#
#             # --- CONSUMER LOOP ---
#             # Process results as they finish (this handles the progress bar too!)
#             completed = 0
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#
#                     # Route the result to the right storage based on ROI index
#                     f_idx = result['frame_idx']
#                     if result['roi_idx'] == 0:
#                         width_results[f_idx] = result['mask_data']
#                     else:
#                         thickness_results[f_idx] = result['mask_data']
#
#                     completed += 1
#
#                     # Update progress (divide by 2 because there are 2 ROIs per frame)
#                     if completed % 2 == 0:
#                         frames_processed = completed / 2
#                         # Start at 50, and add the remaining 50% based on completion
#                         progress_pct = 50 + int((frames_processed / total_frames) * 50)
#                         signals.progress.emit(progress_pct)
#
#                 except Exception as exc:
#                     print(f"Worker generated an exception: {exc}")
#
#     # Reconstruct the ordered lists from the dictionaries
#     width_masks_out = [width_results[i] for i in range(total_frames)]
#     thickness_masks_out = [thickness_results[i] for i in range(total_frames)]
#
#     return {
#         'first_masks': width_masks_out,
#         'second_masks': thickness_masks_out
#     }

def geometry_worker(signals, config: dict):
    file_path = config['file_path']
    distances = config['distances']
    frame_indices = np.asarray(config.get('frame_indices', np.arange(len(distances))), dtype=np.int64)
    min_dist = config['min_dist']
    max_dist = config['max_dist']
    roi_data = config['roi_data']

    mu = config['mu']
    gamma = config['gamma']
    lambda1 = config['lambda1']

    frames_out = []
    width_masks_out, thickness_masks_out = [], []

    total_frames = len(distances)
    if len(frame_indices) != total_frames:
        raise ValueError("Geometry distances and source-frame indices must have equal lengths.")
    if total_frames == 0:
        raise ValueError("Geometry cannot run without any retained frames.")
    if max_dist == min_dist:
        raise ValueError(
            "Geometry cannot interpolate ROIs because all retained distance values are identical."
        )

    # Get optimal core count (leave one or two for the OS/GUI)
    max_workers = max(1, multiprocessing.cpu_count() - 2)

    width_results = {}
    thickness_results = {}

    # Start the multiprocessing pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        signals.message.emit("Reading image data and queuing tasks...")

        # --- PRODUCER LOOP ---
        # Zip the distances array with our new unified frame generator
        for i, (distance, frame) in enumerate(zip(distances, iter_selected_frames(file_path, frame_indices))):

            # 1. Calculate percentage and clamp it mathematically
            pct = (distance - min_dist) / (max_dist - min_dist)
            pct = np.clip(pct, 0.0, 1.0)

            # 2. Get interpolated boxes and seeds
            interp_rois = _interpolate_rois_worker(roi_data, pct)

            # The frame format check remains exactly the same
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
            gray = gray.astype(np.uint8)
            dimensions = []  # Will hold [width_px, length_px]

            for roi_idx, roi in enumerate(interp_rois):
                # 1. CROP THE IMAGE HERE (In the Manager)
                x, y, w, h = roi['roi_rect']
                crop = gray[y: y + h, x: x + w]

                # 2. GENERATE SEED MASK HERE
                if crop.size == 0:
                    raise ValueError(
                        f"Geometry frame {i}, ROI {roi_idx + 1} produced an empty image crop."
                    )

                # Generate Seed Mask
                mask_shape = crop.shape
                seed_mask = np.zeros(mask_shape, dtype=bool)
                coords = roi['seed_coords']

                if not roi.get('seed_shape_type') or not coords:
                    raise ValueError(
                        f"Geometry frame {i}, ROI {roi_idx + 1} is missing a seed shape."
                    )

                if roi['seed_shape_type'] == 'rect':
                    start = (coords['y'], coords['x'])
                    end = (coords['y'] + coords['h'], coords['x'] + coords['w'])
                    rr, cc = rectangle(start=start, end=end, shape=mask_shape)
                    seed_mask[rr, cc] = True
                elif roi['seed_shape_type'] == 'ellipse':
                    rr, cc = ellipse(coords['center_y'], coords['center_x'],
                                     coords['radius_y'], coords['radius_x'], shape=mask_shape)
                    seed_mask[rr, cc] = True

                # 3. PACKAGE THE TINY PAYLOAD
                payload = {
                    'crop': crop,
                    'seed_mask': seed_mask,
                    'mu': mu, 'gamma': gamma, 'lambda1': lambda1,
                    'frame_idx': i,
                    'roi_idx': roi_idx
                }

                # 4. SUBMIT TO WORKER POOL
                future = executor.submit(compute_chan_vese_worker, payload)
                futures.append(future)

                progress_pct = int(((i + 1) / total_frames) * 50)
                signals.progress.emit(progress_pct)

        signals.message.emit("Processing segmentation masks...")

        # --- CONSUMER LOOP ---
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()

                # Route the result to the right storage based on ROI index
                f_idx = result['frame_idx']
                if result['roi_idx'] == 0:
                    width_results[f_idx] = result['mask_data']
                else:
                    thickness_results[f_idx] = result['mask_data']

                completed += 1

                if completed % 2 == 0:
                    frames_processed = completed / 2
                    progress_pct = 50 + int((frames_processed / total_frames) * 50)
                    signals.progress.emit(progress_pct)

            except Exception as exc:
                raise RuntimeError(f"Segmentation worker failed: {exc}") from exc

    # Reconstruct the ordered lists from the dictionaries
    missing_width = [i for i in range(total_frames) if i not in width_results]
    missing_thickness = [i for i in range(total_frames) if i not in thickness_results]
    if missing_width or missing_thickness:
        raise RuntimeError(
            "Geometry processing did not return every expected segmentation result."
        )

    width_masks_out = [width_results[i] for i in range(total_frames)]
    thickness_masks_out = [thickness_results[i] for i in range(total_frames)]

    for i, (width_item, thickness_item) in enumerate(
            zip(width_masks_out, thickness_masks_out)
    ):
        validate_segmentation_mask(
            width_item['mask'],
            f"Geometry frame {i} width segmentation",
        )
        validate_segmentation_mask(
            thickness_item['mask'],
            f"Geometry frame {i} thickness segmentation",
        )

    return {
        'first_masks': width_masks_out,
        'second_masks': thickness_masks_out,
        'trim_revision': config.get('trim_revision', 0),
    }

def keep_largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Keep only the largest connected foreground object in a binary 2D mask.

    Parameters
    ----------
    mask : np.ndarray
        2D binary array where background is 0 and foreground is nonzero/1.
    connectivity : int
        4 or 8. Use 8 if diagonally-touching pixels should count as connected.

    Returns
    -------
    largest_mask : np.ndarray
        Binary 0/1 mask containing only the largest connected foreground object.
    """

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    # Ensure binary uint8 for OpenCV
    binary = (mask > 0).astype(np.uint8)

    # labels has same shape as mask.
    # label 0 is background.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=connectivity
    )

    # If there are no foreground objects, return an empty mask
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    # stats[:, cv2.CC_STAT_AREA] gives area of each label
    # Ignore label 0 because that is background
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    # +1 because we ignored background label 0
    largest_label = 1 + np.argmax(component_areas)

    largest_mask = (labels == largest_label).astype(np.uint8)

    return largest_mask

def compute_chan_vese_worker(payload: dict):
    """
    Expects a payload with:
    - crop: the cropped numpy array
    - seed_mask: the generated seed mask
    - mu, gamma, lambda1
    - frame_idx, roi_idx (to keep track of results since processes finish out of order)
    """
    # 1. Unpack
    crop = payload['crop']
    seed_mask = payload['seed_mask']
    mu, gamma, lambda1 = payload['mu'], payload['gamma'], payload['lambda1']
    frame_idx, roi_idx = payload['frame_idx'], payload['roi_idx']

    if crop.size == 0:
        return {'frame_idx': frame_idx, 'roi_idx': roi_idx,
                'mask_data': {'mask': np.array([]), 'offset_x': 0, 'offset_y': 0}}

    # 2. Apply Gamma & Chan-Vese
    img_float = img_as_float(crop)
    if gamma != 1.0:
        img_float = img_float ** gamma

    cv_result = chan_vese(img_float, mu=mu, lambda1=lambda1, lambda2=1.0,
                          tol=1e-3, max_num_iter=100, dt=0.5,
                          init_level_set=seed_mask, extended_output=True)

    # 3. Morphology & Tight Masking (keep your existing cleanup logic here)
    final_mask = cv_result[0].astype(np.uint8)

    # Optional Morphology cleanup (ensure mask is 0/255 for cv2)
    cleaned_mask = cv2.morphologyEx(final_mask * 255, cv2.MORPH_OPEN, KERNEL)
    binary_mask = cleaned_mask // 255

    binary_mask = keep_largest_component(binary_mask)

    # --- MASK CROPPING & STORAGE ---
    # Find coordinates of all non-zero pixels
    y_idx, x_idx = np.nonzero(binary_mask)

    if len(y_idx) > 0:
        # Get bounding box of the active mask
        min_y, max_y = y_idx.min(), y_idx.max()
        min_x, max_x = x_idx.min(), x_idx.max()

        # Slice the array and cast to boolean (1 byte per pixel)
        tight_mask = binary_mask[min_y:max_y + 1, min_x:max_x + 1].astype(bool)
        mask_data = {
            'mask': tight_mask,
            'offset_x': int(min_x),
            'offset_y': int(min_y)
        }
    else:
        mask_data = {
            'mask': np.array([], dtype=bool),
            'offset_x': 0,
            'offset_y': 0
        }
        # -------------------------------

    # Return the lightweight data
    return {
        'frame_idx': frame_idx,
        'roi_idx': roi_idx,
        'mask_data': mask_data  # dict with 'mask', 'offset_x', 'offset_y'
    }


def validate_segmentation_mask(mask, context="Segmentation mask"):
    """Return a valid 2D mask or stop before downstream geometry math runs."""
    mask = np.asarray(mask)
    if mask.size == 0:
        raise ValueError(f"{context} is empty.")
    if mask.ndim != 2:
        raise ValueError(f"{context} must be two-dimensional; received shape {mask.shape}.")
    if not np.any(mask):
        raise ValueError(f"{context} contains no segmented object.")
    return mask


def _normalize_and_format_mask(mask):
    """Standalone helper for formatting the mask before writing."""
    mask = validate_segmentation_mask(mask, "Threshold video mask")
    if mask.max() <= 1.0 and mask.max() > 0:
        mask = (mask * 255.0)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def video_compiler_worker(signals, config: dict):
    """Background worker for compiling the thresholded video."""
    filepath = config['filepath']
    fps = config['fps']
    first_segments = config['first_segments']
    second_segments = config['second_segments']

    if not first_segments or len(first_segments) != len(second_segments):
        raise ValueError(
            "Threshold video requires two aligned, non-empty segmentation sequences."
        )

    signals.message.emit("Compiling threshold video...")

    max_h1, max_h2, max_w = 0, 0, 0
    for i, (f_item, s_item) in enumerate(zip(first_segments, second_segments)):
        first_mask = validate_segmentation_mask(
            f_item['mask'], f"Threshold video frame {i} first segmentation"
        )
        second_mask = validate_segmentation_mask(
            s_item['mask'], f"Threshold video frame {i} second segmentation"
        )
        h1, w1 = first_mask.shape
        h2, w2 = second_mask.shape
        if h1 > max_h1: max_h1 = h1
        if h2 > max_h2: max_h2 = h2
        if w1 > max_w: max_w = w1
        if w2 > max_w: max_w = w2

    frame_width = max_w
    frame_height = max_h1 + max_h2
    size = (frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(filepath), fourcc, fps, size, True)

    total_frames = len(first_segments)
    for i, (f_item, s_item) in enumerate(zip(first_segments, second_segments)):
        mask1 = _normalize_and_format_mask(f_item['mask'])
        mask2 = _normalize_and_format_mask(s_item['mask'])

        h1, w1 = mask1.shape[:2]
        h2, w2 = mask2.shape[:2]

        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame[0:h1, 0:w1] = mask1
        frame[max_h1:max_h1 + h2, 0:w2] = mask2

        video.write(frame)

        # Optional: Update progress bar while writing
        if i % 10 == 0:
            signals.progress.emit(int((i / total_frames) * 100))

    video.release()

    # Calculate filesize to send back to the UI
    filesize_bytes = os.path.getsize(filepath)
    filesize_mb = filesize_bytes / (1024 * 1024)

    return {
        'filepath': filepath,
        'filesize_mb': filesize_mb
    }
