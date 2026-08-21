import os
import sys
import json
import logging
from inspect import trace

import numpy as np
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot, QTimer, SignalInstance
#from processing.data_transform import zero_data, smooth_data, label_image, create_visual_from_labels, convert_numpy, restore_numpy, n_closest_numbers
from processing.data_loader import (
    extract_buti_settings,
    frame_loader,
    geometry_worker,
    keep_largest_component,
    validate_segmentation_mask,
    video_compiler_worker,
)
from processing.data_transform import serialize_objects, deserialize_objects
from processing.trimming import map_distance_extrema_to_source
from config import APP_VERSION
from widgets.error_bus import user_error
import cv2
from processing.data_transform import numpy_to_qpixmap

import numpy as np
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.draw import disk, ellipse, rectangle
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress


log = logging.getLogger(__name__)

KERNEL_SIZE = 7
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

class DataPipeline(QObject):
    # --- SIGNALS ---
    state_loaded = Signal()

    data_available = Signal(dict)

    # Plot inputs
    plot_selection_changed = Signal(str)
    cycle_selection_changed = Signal(str)

    trimmed_data_available = Signal(object)  # 'object' is safest for passing numpy arrays
    trim_time_changed = Signal(float)
    trim_range_changed = Signal(float, float)

    # Scale inputs
    known_length_changed = Signal(float)
    pixel_length_changed = Signal(float)
    scale_is_manual_changed = Signal(bool)
    manual_conversion_factor_changed = Signal(float)
    conversion_factor_changed = Signal(float)
    scale_line_coords_changed = Signal(list)

    # Frames
    left_image_changed = Signal(np.ndarray)
    roi_min_image_loaded = Signal(np.ndarray)
    roi_max_image_loaded = Signal(np.ndarray)

    rois_ready = Signal(list)
    cropped_images_ready = Signal(list)
    seed_masks_ready = Signal(dict)
    threshed_images_ready = Signal(list)
    clear_threshold_displays = Signal()

    dimension_images_ready = Signal(np.ndarray, np.ndarray)

    geometry_available = Signal(dict)
    geometry_processing_changed = Signal(bool)
    video_compiled_ready = Signal(str)
    mechanics_available = Signal(dict)

    last_pull_available = Signal(dict)

    voxels_available = Signal(object)

    relaxation_available = Signal(dict)

    # Smoothing
    s_value_changed = Signal(int)




    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_manager = None
        self.VERSION = APP_VERSION
        self.author = None
        self.video_formats = [".tiff", ".tif", ".avi", ".mkv"]

        self.loaded_state = False

        self.platform = 'mac' if sys.platform == 'darwin' else 'win'
        self.ctrl_key = "Cmd" if self.platform == "mac" else "Ctrl"

        self.video = None
        self.frame_count = 0
        self.start = 0
        self.frame_count = 0

        # PLOT TAB
        self.plot_selection: str = "Time vs. Force"
        self.cycle_selection: str = "All Cycles"
        self.trim_time: float = 0.0
        self.trim_start_time: float = 0.0
        self.trim_end_time: float = 0.0
        self.data_trimmed = None
        self.active_frame_indices = np.array([], dtype=np.int64)
        self.min_distance_data_index = 0
        self.max_distance_data_index = 0
        self.trim_revision = 0

        # SCALE TAB
        self.known_length = 0.0
        self.pixel_length = 0.0
        self.scale_is_manual = False
        self.manual_conversion_factor = 0.0
        self.conversion_factor = 0.0  # The final, authoritative value
        self.scale_line_coords = []

        self.left_image: np.ndarray | None = None

        self.data = None
        self.buti_settings = {}
        self.frame_data = {}
        self.roi_data = {
            'min': [],
            'max': []
        }

        self.mu = 0
        self.lambda1 = 0
        self.gamma = 0

        self.xy_roi_idx = 0
        self.thickness_roi_idx = 1

        self.layout = "vertical"
        self.voxel_arrays = []
        self.geometry_in_progress = False

        self.exported_file = None

    @Slot(float, float, object, object)
    def set_trimmed_data(
            self,
            start_time: float,
            end_time: float,
            trimmed_data: np.ndarray,
            source_indices: np.ndarray,
    ):
        """Store and broadcast the active data range and its source-frame mapping."""
        source_indices = np.asarray(source_indices, dtype=np.int64)
        if len(trimmed_data) != len(source_indices):
            raise ValueError("Trimmed data and source-frame indices must have equal lengths.")

        had_active_data = self.data_trimmed is not None and self.data_trimmed.size > 0
        range_changed = had_active_data and (
            start_time != self.trim_start_time
            or end_time != self.trim_end_time
            or not np.array_equal(source_indices, self.active_frame_indices)
            or not np.array_equal(trimmed_data, self.data_trimmed)
        )

        self.trim_start_time = start_time
        self.trim_end_time = end_time
        self.trim_time = end_time  # Legacy end-time alias retained for saved-state compatibility.
        self.data_trimmed = trimmed_data
        self.active_frame_indices = source_indices

        if range_changed:
            self.trim_revision += 1
            self._invalidate_derived_analysis()

        log.info(
            "Data range set from %.6fs to %.6fs (%d samples).",
            self.trim_start_time,
            self.trim_end_time,
            len(self.data_trimmed),
        )

        self._recalculate_roi_indices()

        # Emit signals so other tabs can update
        self.trim_time_changed.emit(self.trim_time)
        self.trim_range_changed.emit(self.trim_start_time, self.trim_end_time)
        self.trimmed_data_available.emit(self.data_trimmed)

    def _invalidate_derived_analysis(self):
        """Discard calculations whose array alignment depends on the active range."""
        had_derived_data = any(
            getattr(self, name, None) is not None
            for name in (
                "geometry_data",
                "mechanics_payload",
                "relaxation_payload",
                "last_pull_stretch",
            )
        )

        for name in ("first_segments", "second_segments", "_raw_geometry", "_smoothed_geometry"):
            if hasattr(self, name):
                delattr(self, name)

        for name in (
            "geometry_data",
            "mechanics_payload",
            "relaxation_payload",
            "last_pull_stretch",
            "last_pull_stress",
            "p_spline",
        ):
            setattr(self, name, None)

        self.voxel_arrays = []

        if had_derived_data and self.task_manager is not None:
            self.task_manager.status_updated.emit(
                "Trim range changed - run Get Geometry again to refresh analysis results."
            )


    ### region HEADER

    def on_author_changed(self, new_author):
        self.author = new_author
        log.info(f"User: {self.author}")

    def load_video_file(self, file_path: str, index=None):
        self.video = file_path
        if index is None:
            index = self.start
        self.task_manager.queue_task(
            frame_loader,  # function
            self.video,  # file_path
            [index],  # frame_indices
            True,  # return frame_count
            on_result=self.initial_frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def initial_frame_loaded(self, result: dict):
        """
        Callback function executed when the frame_loader task is complete.
        'result' is the dictionary of NumPy arrays returned by frame_loader.
        """
        log.info(f"Received {len(result)-2} loaded frames from worker.")
        if not result:
            log.warning("Frame loader returned no frames.")
            return

        it = iter(result)
        first_frame_index = next(it)
        self.left_image = result[first_frame_index]

        if self.left_image is None:
            log.error("Did not receive left frame")
        self.left_image_changed.emit(self.left_image)

        self.frame_count = next(it)
        log.info(f"Frame count found: {self.frame_count}")

        # Safely handle data: TIFF embedded data will be here, MKV data won't.
        worker_data = result.get("data")
        if worker_data is not None:
            log.info("Using embedded data from frame loader.")
            self.data = worker_data

        # Failsafe check
        if not hasattr(self, "data") or not self.data:
            log.error("No telemetry data available! Pipeline cannot proceed.")
            return

        self.buti_settings = extract_buti_settings(self.data)
        if self.buti_settings:
            log.info("Loaded BUTI acquisition settings: %s", self.buti_settings)

        # From here down, self.data is guaranteed to exist and have the right columns
        log.info(f"Data keys available: {self.data.keys()}")
        self.max_distance_data_index = int(np.argmax(self.data["distance"]))
        self.min_distance_data_index = int(np.argmin(self.data["distance"]))
        self.max_distance_index = self.max_distance_data_index
        self.min_distance_index = self.min_distance_data_index
        self.data_available.emit(self.data)
        if len([i for i, x in enumerate(self.data["cycle"]) if x == np.max(self.data['cycle'])]) < 5:
            self.last_cycle = np.max(self.data["cycle"]) - 1
        else:
            self.last_cycle = np.max(self.data["cycle"])
        print("Last cycle:", self.last_cycle)
        print("Hz:", np.mean(np.gradient(self.data["time_s"])))
        self.fps = int(round(1/np.mean(np.gradient(self.data["time_s"]))))
        print("FPS:", self.fps)
        self.load_frames([self.max_distance_index, self.min_distance_index])

    def load_frames(self, index):
        self.task_manager.queue_task(
            frame_loader,  # function
            self.video,  # file_path
            index,  # frame_indices
            False,  # return frame_count
            on_result=self.frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def frame_loaded(self, result):
        for k, v in result.items():
            self.frame_data[k] = v
            if k == self.max_distance_index:
                print("loaded max roi frame")
                self.roi_max_image_loaded.emit(v)
            if k == self.min_distance_index:
                print('loaded min roi frame')
                self.roi_min_image_loaded.emit(v)

    # endregion

    ### region PLOT TAB

    @Slot(str)
    def set_plot_selection(self, selection_text: str):
        """Sets the active plot selection (e.g., 'Time vs. Force')"""
        if self.plot_selection != selection_text:
            self.plot_selection = selection_text
            self.plot_selection_changed.emit(self.plot_selection)

    @Slot(str)
    def set_cycle_selection(self, selection_text: str):
        """Sets the active cycle filter (e.g., 'All Cycles')"""
        if self.cycle_selection != selection_text:
            self.cycle_selection = selection_text
            self.cycle_selection_changed.emit(self.cycle_selection)

    # endregion

    ### region SCALE TAB

    def _recalculate_conversion_factor(self):
        """Central calculation. Called whenever an input changes."""
        new_factor = 0.0
        if self.scale_is_manual:
            new_factor = self.manual_conversion_factor
        elif self.known_length > 0 and self.pixel_length > 0:
            new_factor = self.pixel_length / self.known_length

        # Use the main setter to update the value and emit the signal
        self.set_conversion_factor(new_factor, force_update=True)

    def set_known_length(self, length: float):
        if self.known_length != length:
            self.known_length = length
            self.known_length_changed.emit(self.known_length)
            self._recalculate_conversion_factor()

    def set_scale_line_data(self, flat_data: list):
        # Guard clause in case something weird happens
        if not flat_data or len(flat_data) != 5:
            return

        # Unpack the 5 numbers
        length = flat_data[0]

        # Reconstruct the coordinate list for saving/injecting later
        coords = [
            (flat_data[1], flat_data[2]),
            (flat_data[3], flat_data[4])
        ]

        self.scale_line_coords = coords

        if self.pixel_length != length:
            self.pixel_length = length
            self.pixel_length_changed.emit(self.pixel_length)
            self._recalculate_conversion_factor()

    def set_scale_is_manual(self, is_manual: bool):
        if self.scale_is_manual != is_manual:
            self.scale_is_manual = is_manual
            self.scale_is_manual_changed.emit(self.scale_is_manual)
            self._recalculate_conversion_factor()

    def set_manual_conversion_factor(self, factor: float):
        """This is called when the user types in the manual spinbox."""
        if self.manual_conversion_factor != factor:
            self.manual_conversion_factor = factor
            # Only recalculate if we are currently in manual mode
            self.manual_conversion_factor_changed.emit(self.manual_conversion_factor)
            if self.scale_is_manual:
                self._recalculate_conversion_factor()

    def set_conversion_factor(self, factor: float, force_update=False):
        """
        This is the final setter for the authoritative value.
        It's called by the recalculate method or can be set directly.
        """
        if self.conversion_factor != factor or force_update:
            self.conversion_factor = factor
            log.info(f"Conversion factor updated: {self.conversion_factor}")
            #self.data_version += 1
            self.conversion_factor_changed.emit(self.conversion_factor)

    # endregion

    ### region ROI TAB

    def _recalculate_roi_indices(self):
        """Calculates min/max distance indices from the active (trimmed) dataset."""
        if self.data_trimmed is None or self.data_trimmed.size == 0:
            return

        if len(self.active_frame_indices) != len(self.data_trimmed):
            log.warning("Active frame mapping was missing or stale; rebuilding it from row positions.")
            self.active_frame_indices = np.arange(len(self.data_trimmed), dtype=np.int64)

        # Local indices address active telemetry arrays; source indices address video frames.
        (
            new_max_data_idx,
            new_min_data_idx,
            new_max_frame_idx,
            new_min_frame_idx,
        ) = map_distance_extrema_to_source(self.data_trimmed, self.active_frame_indices)

        self.max_distance_data_index = new_max_data_idx
        self.min_distance_data_index = new_min_data_idx

        frames_to_load = []

        # The ROI image cache is keyed by absolute source-frame index.
        if new_max_frame_idx != self.max_distance_index:
            log.info(f"Max distance frame changed from {self.max_distance_index} to {new_max_frame_idx}")
            self.max_distance_index = new_max_frame_idx
            frames_to_load.append(self.max_distance_index)

        if new_min_frame_idx != self.min_distance_index:
            log.info(f"Min distance frame changed from {self.min_distance_index} to {new_min_frame_idx}")
            self.min_distance_index = new_min_frame_idx
            frames_to_load.append(self.min_distance_index)

        # If either index changed, send them to the worker to load the new frames.
        # Once loaded, your existing `frame_loaded` callback will catch them
        # and emit the image signals to update the UI automatically.
        if frames_to_load:
            log.info(f"Loading new ROI frames: {frames_to_load}")
            self.load_frames(frames_to_load)

    def _generate_crops_for_target(self, target):
        """Helper: Slices the arrays and stores them in the roi_data dictionary."""
        if not self.frame_data:
            return

        if target == 'min':
            image = self.frame_data.get(self.min_distance_index)  # Get full-size image
        else:
            image = self.frame_data.get(self.max_distance_index)

        if image is None:
            return

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply crops
        for roi_dict in self.roi_data[target]:
            rect = roi_dict["roi_rect"]
            x, y, w, h = roi_dict["roi_rect"]

            if w > 0 and h > 0:
                crop = gray[y:y + h, x:x + w]
                if crop.size > 0:
                    roi_dict["crop_img"] = crop

    def receive_roi_data(self, roi_data, target):
        """
        Receives QRects from ROITab, initializes the nested structure,
        generates crops, and emits QPixmaps for the SeedTab.
        """
        print(f"{len(roi_data)} boxes for {target}: {roi_data}")

        # 1. Reset and populate the list of dictionaries for this target
        self.roi_data[target] = []
        for rect in roi_data:
            rect_tuple = (rect.x(), rect.y(), rect.width(), rect.height())
            self.roi_data[target].append({
                "roi_rect": rect_tuple,
                "crop_img": None,
                "seed_shape_type": None,
                "seed_coords": None,
                "seed_mask": None
            })


        self.correlate_rois()

        # 2. Generate crops and store them directly inside the nested dict
        self._generate_crops_for_target(target)

        self.clear_threshold_displays.emit()

        # 3. Gather all valid crops in order (Min first, then Max) to send to UI
        crops_pixmaps = []
        for t in ['min', 'max']:
            for roi_dict in self.roi_data.get(t, []):
                if roi_dict["crop_img"] is not None:
                    crops_pixmaps.append(numpy_to_qpixmap(roi_dict["crop_img"]))

        # 4. Emit the Pixmaps to the Seed Tab
        if crops_pixmaps:
            self.cropped_images_ready.emit(crops_pixmaps)

    # endregion

    ### region SEED TAB

    def receive_seed_shape(self, index, shape_type, data):
        """
        Called by UI when user finishes drawing. Maps the UI index back
        to the nested dictionary and stores the seed data/mask.
        """
        # 1. Map the flat UI index (0, 1, 2, 3) to target ('min' or 'max') and sub-index
        target = None
        sub_index = 0
        min_len = len(self.roi_data.get('min', []))
        max_len = len(self.roi_data.get('max', []))

        if index < min_len:
            target = 'min'
            sub_index = index
        elif index < min_len + max_len:
            target = 'max'
            sub_index = index - min_len
        else:
            return  # Index out of bounds (user clicked an empty canvas)

        # 2. Grab the specific dictionary for this ROI
        roi_dict = self.roi_data[target][sub_index]
        crop_img = roi_dict.get("crop_img")

        if crop_img is None:
            return

        # 3. Handle clearing the shape (Undo/Reset)
        if shape_type is None or not data:
            roi_dict["seed_shape_type"] = None
            roi_dict["seed_coords"] = None
            roi_dict["seed_mask"] = None
            print(f"Mask cleared for image {index} ({target} ROI {sub_index})")
            self.clear_threshold_displays.emit()
            return

        # 4. Generate the Mask
        image_shape = crop_img.shape
        mask = np.zeros(image_shape, dtype=bool)

        if shape_type == 'rect':
            start = (data['y'], data['x'])
            end = (data['y'] + data['h'], data['x'] + data['w'])
            rr, cc = rectangle(start=start, end=end, shape=image_shape)
            mask[rr, cc] = True

        elif shape_type == 'ellipse':
            rr, cc = ellipse(data['center_y'], data['center_x'],
                             data['radius_y'], data['radius_x'], shape=image_shape)
            mask[rr, cc] = True

        # 5. Save everything into the unified dictionary
        roi_dict["seed_shape_type"] = shape_type
        roi_dict["seed_coords"] = data
        roi_dict["seed_mask"] = mask
        print(f"Mask created for image {index} ({target} ROI {sub_index}) with shape {shape_type}")

        # 6. Check if we have all required masks to emit
        # (Extracts non-None masks from the nested structure)
        valid_masks = [
            item["seed_mask"]
            for t in ['min', 'max']
            for item in self.roi_data.get(t, [])
            if item.get("seed_mask") is not None
        ]

        if len(valid_masks) == 4:
            self.seed_masks_ready.emit(valid_masks)
        else:
            # CASCADE: If we have less than 4 masks, we can't threshold.
            self.clear_threshold_displays.emit()

    # endregion

    ### region THRESHOLD TAB

    def run_chan_vese(self, img, mu, gamma, lambda1, seed):
        image = img_as_float(img)
        if gamma != 1.0:
            image = image ** gamma

        cv_result = chan_vese(image,
                              mu=mu,
                              lambda1=lambda1,
                              lambda2=1.0,
                              tol=1e-3,
                              max_num_iter=100,
                              dt=0.5,
                              init_level_set=seed,
                              extended_output=True)

        final_mask = cv_result[0]
        # final_energies = cv_result[2]  # To check convergence
        raw_mask = final_mask.astype(np.uint8) * 255
        cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, KERNEL)

        cleaned_mask = keep_largest_component(cleaned_mask) * 255

        return cleaned_mask

    def apply_threshold(self, smoothness_val: int, shadow_recovery_val: int):
        """
        Applies Chan-Vese segmentation to all valid crops using their paired seed masks.
        """
        processed_pixmaps = []

        # 1. Gather all valid ROI dictionaries that have both a crop and a mask
        valid_rois = []
        for target in ['min', 'max']:
            for roi_dict in self.roi_data.get(target, []):
                crop = roi_dict.get("crop_img")
                mask = roi_dict.get("seed_mask")

                if crop is not None and mask is not None:
                    valid_rois.append(roi_dict)

        # Ensure we have all 4 before doing heavy computation
        if len(valid_rois) != 4:
            print("Waiting for all 4 seed masks to be drawn...")
            return

        # 2. Process each valid ROI
        for roi_dict in valid_rois:
            img = roi_dict["crop_img"]
            seed_mask = roi_dict["seed_mask"]

            # Convert to float for Chan-Vese (skimage handles the 0.0-1.0 mapping)

            mu_norm = (smoothness_val - 1) / 99.0
            mu = 0.001 + (mu_norm ** 2) * (0.2 - 0.001)

            # 2. Calculate Coupled Shadow Parameters (Linear scale)
            # shadow_recovery_val is 0 to 100
            shadow_norm = shadow_recovery_val / 100.0

            # Map 0->100 to Gamma 1.0->0.5
            gamma_val = 1.0 - (shadow_norm * 0.5)

            # Map 0->100 to Lambda1 1.0->0.2
            lambda1_val = 1.0 - (shadow_norm * 0.8)

            cleaned_mask = self.run_chan_vese(img, mu, gamma_val, lambda1_val, seed_mask)

            # --- OVERLAY LOGIC ---
            # 1. Ensure the original crop is 3-channel (BGR) so we can apply color
            if len(img.shape) == 2:
                base_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                base_bgr = img.copy()

            # 2. Define the overlay color and transparency level
            overlay_color = (0, 255, 0)  # BGR format: Green (Change to 0,0,255 for Red)
            alpha = 0.4  # Transparency (0.0 is invisible, 1.0 is a solid block of color)

            # 3. Create a copy of the base image and paint the mask area solid green
            color_layer = base_bgr.copy()
            color_layer[cleaned_mask == 255] = overlay_color

            # 4. Blend the solid color layer with the original image
            blended_output = cv2.addWeighted(color_layer, alpha, base_bgr, 1 - alpha, 0)

            # 5. Convert to QPixmap (your helper already handles BGR -> RGB conversion)
            pix = numpy_to_qpixmap(blended_output)
            processed_pixmaps.append(pix)

            roi_dict["cv_img"] = cleaned_mask

        self.mu = mu
        self.lambda1 = lambda1_val
        self.gamma_val = gamma_val

        # 5. Emit back to the UI
        self.threshed_images_ready.emit(processed_pixmaps)

    # endregion

    ### region GEOMETRY IMAGES

    def request_dimension_images(self):
        """
        Called by the GeometryTab when it opens or when dimensions are swapped.
        Fetches the 'min' state crops based on current index tracking and emits them.
        """
        min_rois = self.roi_data.get('min', [])
        if len(min_rois) >= 2:
            w_img = min_rois[self.xy_roi_idx].get('cv_img')
            t_img = min_rois[self.thickness_roi_idx].get('cv_img')

            if w_img is not None and t_img is not None:
                self.dimension_images_ready.emit(w_img, t_img)

    def swap_dimensions(self):
        """
        Flips the state mapping for Width and Thickness, then updates the UI.
        """
        # Swap the tracking indices
        self.xy_roi_idx, self.thickness_roi_idx = self.thickness_roi_idx, self.xy_roi_idx
        print(f"Swapped Dimensions! Width is now ROI {self.xy_roi_idx}, Thickness is ROI {self.thickness_roi_idx}")

        # Emit the new images to immediately update the visual reference in the UI
        self.request_dimension_images()
        if hasattr(self, 'first_segments'):
            self.calculate_dimensions()

        # NOTE: If we already have computed segmentation masks stored,
        # we will eventually call our math recalculation function right here!

    def correlate_rois(self):
        """
        Pairs the 'min' ROIs with the correct 'max' ROIs based on spatial overlap.
        Reorders the 'max' list in self.roi_data to perfectly match the 'min' list.
        """
        min_data = self.roi_data.get('min', [])
        max_data = self.roi_data.get('max', [])

        # We need exactly 2 ROIs in both states to perform a correlation
        if len(min_data) != 2 or len(max_data) != 2:
            print("Cannot correlate: Need exactly 2 ROIs in both min and max states.")
            return

        m0_rect = min_data[0]["roi_rect"]
        m1_rect = min_data[1]["roi_rect"]

        x0_rect = max_data[0]["roi_rect"]
        x1_rect = max_data[1]["roi_rect"]

        def get_overlap_area(rect1, rect2):
            """Helper to calculate the overlapping area of two (x, y, w, h) tuples."""
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2

            # Find the coordinates of the intersection rectangle
            left = max(x1, x2)
            right = min(x1 + w1, x2 + w2)
            top = max(y1, y2)
            bottom = min(y1 + h1, y2 + h2)

            # Calculate width and height of intersection
            inter_width = right - left
            inter_height = bottom - top

            # If both width and height are positive, they overlap
            if inter_width > 0 and inter_height > 0:
                return inter_width * inter_height

            return 0

        # --- Test Configuration A ---
        # Min 0 pairs with Max 0
        # Min 1 pairs with Max 1
        score_a = get_overlap_area(m0_rect, x0_rect) + get_overlap_area(m1_rect, x1_rect)

        # --- Test Configuration B ---
        # Min 0 pairs with Max 1
        # Min 1 pairs with Max 0
        score_b = get_overlap_area(m0_rect, x1_rect) + get_overlap_area(m1_rect, x0_rect)

        # --- Apply the Best Match ---
        if score_a >= score_b:
            print("Correlation: Min 0 -> Max 0, Min 1 -> Max 1")
            # They are already in the correct order, do nothing.
        else:
            print("Correlation: Min 0 -> Max 1, Min 1 -> Max 0")
            # Swap the max list so the indices match the min list
            self.roi_data['max'] = [max_data[1], max_data[0]]

    def get_interpolated_data(self, pct: float) -> list[dict]:
        """
        Calculates the interpolated ROI bounding boxes and seed shapes between min and max states.
        Returns a list of dictionaries structured identically to the base roi_data items.
        """
        if not self.roi_data.get("min") or not self.roi_data.get("max"):
            return []

        def lerp_int(start: float, end: float, pct: float) -> int:
            """Standard linear interpolation, correctly rounded to the nearest integer pixel."""
            return int(round(start + (end - start) * pct))

        def lerp_rect(r_min: QRect, r_max: QRect, pct: float) -> QRect:
            """Interpolates all four dimensions of a QRect."""
            x = lerp_int(r_min.x(), r_max.x(), pct)
            y = lerp_int(r_min.y(), r_max.y(), pct)
            w = lerp_int(r_min.width(), r_max.width(), pct)
            h = lerp_int(r_min.height(), r_max.height(), pct)
            return QRect(x, y, w, h)

        def lerp_seed(min_dict: dict, max_dict: dict, pct: float) -> tuple:
            """Interpolates the seed coordinates if the shape types match."""
            shape_type = min_dict.get("seed_shape_type")

            # Enforce shape matching (cannot interpolate a rect into an ellipse)
            if not shape_type or shape_type != max_dict.get("seed_shape_type"):
                return None, None

            c_min = min_dict.get("seed_coords")
            c_max = max_dict.get("seed_coords")

            if not c_min or not c_max:
                return None, None

            interp_coords = {}

            if shape_type == 'rect':
                interp_coords['x'] = lerp_int(c_min['x'], c_max['x'], pct)
                interp_coords['y'] = lerp_int(c_min['y'], c_max['y'], pct)
                interp_coords['w'] = lerp_int(c_min['w'], c_max['w'], pct)
                interp_coords['h'] = lerp_int(c_min['h'], c_max['h'], pct)

            elif shape_type == 'ellipse':
                interp_coords['center_x'] = lerp_int(c_min['center_x'], c_max['center_x'], pct)
                interp_coords['center_y'] = lerp_int(c_min['center_y'], c_max['center_y'], pct)
                interp_coords['radius_x'] = lerp_int(c_min['radius_x'], c_max['radius_x'], pct)
                interp_coords['radius_y'] = lerp_int(c_min['radius_y'], c_max['radius_y'], pct)

            return shape_type, interp_coords

        interpolated_rois = []

        # Iterate through the paired ROIs (assuming correlate_rois() was already run)
        for i in range(2):
            min_roi = self.roi_data["min"][i]
            max_roi = self.roi_data["max"][i]

            # 1. Interpolate the global bounding box
            interp_rect = lerp_rect(min_roi["roi_rect"], max_roi["roi_rect"], pct)

            # 2. Interpolate the local seed shape
            shape_type, interp_coords = lerp_seed(min_roi, max_roi, pct)

            # 3. Package it into the standard dictionary format
            interpolated_rois.append({
                "roi_rect": interp_rect,
                "seed_shape_type": shape_type,
                "seed_coords": interp_coords,
                "crop_img": None,  # To be filled when the specific frame is loaded
                "seed_mask": None  # To be generated from the coords
            })

        return interpolated_rois

    def get_geometry(self):
        """
        Packages current state data and kicks off the background calculation task.
        """
        if self.geometry_in_progress:
            self._geometry_warning(
                "Geometry is already processing.",
                "Wait for the current geometry job to finish before starting another one.",
            )
            return False

        if not self.video or self.data is None or "distance" not in self.data:
            self._geometry_warning(
                "Geometry cannot start because data or video is missing.",
                "Import a supported video and its telemetry before running Get Geometry.",
            )
            return False

        if len(self.roi_data.get('min', [])) != 2 or len(self.roi_data.get('max', [])) != 2:
            self._geometry_warning(
                "Geometry requires two ROIs at minimum and maximum distance.",
                "Complete the ROI and Seed tabs before running Get Geometry.",
            )
            return False

        if self.data_trimmed is None or self.data_trimmed.size == 0:
            self._geometry_warning(
                "Geometry cannot start because the active trim range is empty.",
                "Apply a trim range containing at least one telemetry sample.",
            )
            return False

        distances = self.data_trimmed["distance"]
        frame_indices = np.asarray(self.active_frame_indices, dtype=np.int64)
        if len(frame_indices) != len(distances):
            self._geometry_warning(
                "Geometry cannot start because frame mapping is not aligned with telemetry.",
                "Reset the trim range and try again. If the problem continues, reload the video.",
            )
            return False

        if np.max(distances) == np.min(distances):
            self._geometry_warning(
                "Geometry requires a changing distance to interpolate the ROIs.",
                "Choose a wider trim range that contains more than one distance value.",
            )
            return False

        if self.task_manager is None:
            self._geometry_warning(
                "Geometry processing is not available in this session.",
                "Open a new analysis session and reload the video.",
            )
            return False

        # Package a "snapshot" so the thread doesn't read live UI variables
        snapshot_config = {
            'file_path': self.video,
            'distances': distances,
            'frame_indices': frame_indices,
            'min_dist': distances[self.min_distance_data_index],
            'max_dist': distances[self.max_distance_data_index],
            'roi_data': self.roi_data,  # QRects and dicts are safe to pass by reference if we only read them
            'conversion_factor': self.conversion_factor,
            'mu': getattr(self, 'mu', 0.05),  # Fallbacks in case apply_threshold wasn't run
            'gamma': getattr(self, 'gamma_val', 1.0),
            'lambda1': getattr(self, 'lambda1', 1.0),
            'trim_revision': self.trim_revision,
        }

        # Queue the heavy-lifting task
        self._set_geometry_processing(True)
        self.task_manager.queue_task(
            geometry_worker,  # The pure worker function
            snapshot_config,  # The configuration dict
            on_result=self._on_geometry_computed,
            on_error=self._on_geometry_failed,
        )
        return True

    def _geometry_warning(self, message, hint):
        log.warning("%s %s", message, hint)
        if self.task_manager is not None:
            self.task_manager.status_updated.emit(message)
        user_error(message, hint)

    def _set_geometry_processing(self, active):
        active = bool(active)
        if self.geometry_in_progress != active:
            self.geometry_in_progress = active
            self.geometry_processing_changed.emit(active)

    def _on_geometry_failed(self, err_tb):
        exc, _traceback = err_tb
        self._set_geometry_processing(False)
        user_error(
            "Geometry processing stopped without changing analysis results.",
            f"{exc}\n\nReview the ROI and Seed selections, then try again.",
        )
        return True

    def _on_geometry_computed(self, result: dict):
        """
        Callback triggered when the background task finishes successfully.
        """
        if result.get("trim_revision", self.trim_revision) != self.trim_revision:
            log.info("Discarding geometry computed for an outdated trim range.")
            self._set_geometry_processing(False)
            return

        print("Geometry computed successfully. Calculating dimensions...")
        self.first_segments = result["first_masks"]
        self.second_segments = result["second_masks"]
        self.calculate_dimensions()
        # 1. Extract the base filename without extensions or trailing tags
        filename = os.path.splitext(os.path.basename(self.video))[0].strip("recording_")
        filename = filename.replace("_video", "")
        filename = filename.replace("_c","")
        # 2. Define the new cross-platform path: User's Home Directory -> Documents -> Braid
        braid_dir = Path.home() / "Documents" / "Braid"
        # 3. Safely create the directory (does nothing if it already exists)
        braid_dir.mkdir(parents=True, exist_ok=True)
        # 4. Construct the final output filepath
        filepath = braid_dir / f"{filename}_threshold.mp4"
        # 5. Compile the video
        # Package config for the background task
        snapshot_config = {
            'filepath': filepath,
            'fps': getattr(self, 'fps', 20),
            'first_segments': self.first_segments,
            'second_segments': self.second_segments
        }

        # Queue the video writer
        self.task_manager.queue_task(
            video_compiler_worker,
            snapshot_config,
            on_result=self._on_video_compiled,
            on_error=self._on_geometry_failed,
        )

    def _on_video_compiled(self, result: dict):
        """Callback for when the video background task completes."""
        path = result['filepath']
        size_mb = result['filesize_mb']
        # Construct the friendly message
        msg = f"Ready - Finished writing threshed video to {path} ({size_mb:.2f} MB)"
        # Delay the emission by 100ms so it overwrites the TaskManager's default 'Ready' reset
        QTimer.singleShot(100, lambda: self.task_manager.status_updated.emit(msg))
        print(f"Video compiled successfully: {msg}")

        directory_path = str(Path(path).parent)
        self.video_compiled_ready.emit(directory_path)
        self._set_geometry_processing(False)


    # endregion

    # region DIMENSIONAL DATA GENERATION

    def calculate_dimensions(self):
        """
        Calculates X, Y, and Z dimensions from the stored segmentation masks.
        Calculates the true Y-Z cross-sectional area (3D Voxels).
        Caches both raw and smoothed data for instant UI toggling.
        """
        if not hasattr(self, 'first_segments') or not hasattr(self, 'second_segments'):
            print("No segmentation data available to calculate dimensions.")
            return

        print("Calculating XYZ dimensions from stored masks...")

        frames = []
        raw_x = []
        raw_y = []
        raw_z = []
        self.voxel_arrays = []
        areas = []
        min_areas = []
        volumes = []

        if self.xy_roi_idx == 0:
            xy_masks_list = self.first_segments
            z_masks_list = self.second_segments
        else:
            xy_masks_list = self.second_segments
            z_masks_list = self.first_segments

        num_frames = len(xy_masks_list)
        if num_frames == 0 or num_frames != len(z_masks_list):
            raise ValueError(
                "Geometry requires two aligned, non-empty segmentation sequences."
            )
        cf = self.conversion_factor if self.conversion_factor > 0 else 1.0

        for i in range(num_frames):
            frames.append(i)

            # --- XY Image Processing ---
            xy_mask = validate_segmentation_mask(
                xy_masks_list[i]['mask'],
                f"Geometry frame {i} width segmentation",
            )
            if xy_mask.size > 0:
                x_sums = np.sum(xy_mask, axis=1)
                valid_x = x_sums[x_sums > 0]
                raw_x.append(np.max(valid_x) if len(valid_x) > 0 else 0)

                y_sums = np.sum(xy_mask, axis=0)
                valid_y = y_sums[y_sums > 0]
                raw_y.append(np.mean(valid_y) if len(valid_y) > 0 else 0)
            else:
                raw_x.append(0)
                raw_y.append(0)

            # --- Z Image Processing ---
            z_mask = validate_segmentation_mask(
                z_masks_list[i]['mask'],
                f"Geometry frame {i} thickness segmentation",
            )
            if z_mask.size > 0:
                z_sums = np.sum(z_mask, axis=0)
                valid_z = z_sums[z_sums > 0]
                raw_z.append(np.mean(valid_z) if len(valid_z) > 0 else 0)
            else:
                raw_z.append(0)

            h, w = xy_mask.shape
            h2, w2 = z_mask.shape
            new_h = int(round((w / w2) * h2))
            xy_mask_bool = xy_mask.astype(bool)

            z_mask_scaled = cv2.resize(
                z_mask.astype(np.uint8),
                (w, new_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            xyz_mask = np.zeros((h, w, new_h), dtype=bool)
            xyz_mask[:] = xy_mask_bool[:, :, None]
            xyz_mask &= z_mask_scaled.T[None, :, :]
            self.voxel_arrays.append(xyz_mask)
            # --- Cross-Sectional Area Calculations ---
            # 1D array of areas for each slice along the stretch axis
            slice_areas = np.sum(xyz_mask, axis=(0, 2))

            # 1. Mean Area (Original)
            areas.append(np.mean(slice_areas))
            #min_areas.append(np.min(np.sum(xyz_mask, axis=(0, 2)))) # Poor min method, fixates on end effects
            # 2. Min Area (Middle 50% Spatial Slice)
            active_indices = np.nonzero(slice_areas)[0]

            if len(active_indices) > 0:
                first_idx = active_indices[0]
                last_idx = active_indices[-1]
                active_length = last_idx - first_idx + 1

                # Shave 25% off both the front and the back to look only at the middle 50%
                start_trim = first_idx + int(active_length * 0.25)
                end_trim = first_idx + int(active_length * 0.75)

                # Safeguard in case the object is only a few pixels long
                if start_trim < end_trim:
                    trimmed_areas = slice_areas[start_trim:end_trim]
                    min_areas.append(np.min(trimmed_areas))
                else:
                    # Fallback to the absolute min of the active volume if it's super short
                    min_areas.append(np.min(slice_areas[active_indices]))
            else:
                min_areas.append(0)

            volumes.append(np.sum(xyz_mask))

        areas_mm = np.array(areas) / (cf ** 2)
        min_areas_mm = np.array(min_areas) / (cf ** 2)
        volumes_mm3 = np.array(volumes) / (cf ** 3)
        self.voxels_available.emit(self.voxel_arrays)

        # Convert to real-world units (mm)
        x_mm = np.array(raw_x) / cf
        y_mm = np.array(raw_y) / cf
        z_mm = np.array(raw_z) / cf

        # --- Store Raw Data ---
        self._raw_geometry = {
            'frames': frames,
            'dim_x': x_mm.tolist(),
            'dim_y': y_mm.tolist(),
            'dim_z': z_mm.tolist(),
            'area': areas_mm.tolist(),
            'min_area': min_areas_mm.tolist(),
            'volume': volumes_mm3.tolist()
        }

        # --- Setup & Apply Smoothing ---
        calc_window = int(len(frames) * 0.05)
        window_length = max(5, calc_window if calc_window % 2 != 0 else calc_window + 1)

        if window_length > 3 and len(frames) >= window_length:
            from scipy.signal import savgol_filter
            smooth_x = savgol_filter(x_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_y = savgol_filter(y_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_z = savgol_filter(z_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_area = savgol_filter(areas_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_min_area = savgol_filter(min_areas_mm, window_length=window_length, polyorder=3, mode='nearest')
            smooth_volume = savgol_filter(volumes_mm3, window_length=window_length, polyorder=3, mode='nearest')
        else:
            smooth_x = x_mm
            smooth_y = y_mm
            smooth_z = z_mm
            smooth_area = areas_mm
            smooth_min_area = min_areas_mm
            smooth_volume = volumes_mm3

        # --- Store Smoothed Data ---
        self._smoothed_geometry = {
            'frames': frames,
            'dim_x': smooth_x.tolist(),
            'dim_y': smooth_y.tolist(),
            'dim_z': smooth_z.tolist(),
            'area': smooth_area.tolist(),
            'min_area': smooth_min_area.tolist(),
            'volume': smooth_volume.tolist()
        }

        print("XYZ Dimensions calculated. Dispatching data...")
        self._dispatch_geometry()

    @Slot(bool)
    def set_smoothing_enabled(self, enabled: bool):
        """Toggles smoothing without recalculating image masks."""
        self.use_smoothing = enabled
        print(f"Geometry smoothing enabled: {self.use_smoothing}")
        # Only dispatch if we actually have data stored
        if hasattr(self, '_raw_geometry') and hasattr(self, '_smoothed_geometry'):
            self._dispatch_geometry()

    def _dispatch_geometry(self):
        """Selects the correct dataset and pushes it to the UI and downstream tabs."""
        if getattr(self, 'use_smoothing', True):
            self.geometry_data = self._smoothed_geometry
        else:
            self.geometry_data = self._raw_geometry

        self.geometry_available.emit(self.geometry_data)

        # Cascade the new data through the remaining tabs
        self.calculate_mechanics()
        self.calculate_relaxation()

    # endregion

    def calculate_mechanics(self):
        """
        Derives True Stress, Strain, Poisson's Ratios, and Energy Dissipation
        using strict X, Y, Z coordinate mapping.
        """
        print("Calculating biomechanics...")
        if getattr(self, 'geometry_data', None) is None or self.data is None:
            print("Cannot calculate mechanics: Missing geometry or machine data.")
            return

        # 1. Gather Arrays
        geom = self.geometry_data
        dim_x_opt = np.array(geom['dim_x'])  # Optical transverse loading direction
        dim_y = np.array(geom['dim_y'])  # Optical longitudinal unloaded direction
        dim_z = np.array(geom['dim_z'])  # Optical transverse unloaded direction
        area = np.array(geom['area'])  # Cross-section normal to X (Y * Z)
        min_area = np.array(geom['min_area'])  # Minimum cross-section (Bottleneck)
        volume = np.array(geom['volume'])  # Total 3D voxel volume
        force_mN = self.data_trimmed['force']
        if "2026-02-05" in str(self.video):  # Quirk for backwards load cell data
            force_mN = -force_mN
        machine_x = self.data_trimmed['distance']  # Mechanical loading direction
        cycle_flags = self.data_trimmed['cycle']
        time_s = self.data_trimmed['time_s']

        # Failsafe: Ensure arrays are perfectly aligned in length
        min_len = min(len(dim_x_opt), len(force_mN))
        if min_len == 0:
            return

        dim_x_opt = dim_x_opt[:min_len]
        dim_y = dim_y[:min_len]
        dim_z = dim_z[:min_len]
        area = area[:min_len]
        min_area = min_area[:min_len]
        volume = volume[:min_len]

        force_mN = force_mN[:min_len]
        machine_x = machine_x[:min_len]
        cycle_flags = cycle_flags[:min_len]
        time_s = time_s[:min_len]

        # 2. Establish Rest State (t=0 Baselines)
        X0_opt = np.mean(dim_x_opt[:3]) if np.mean(dim_x_opt[:3]) != 0 else 1e-6
        Y0 = np.mean(dim_y[:3]) if np.mean(dim_y[:3]) != 0 else 1e-6
        Z0 = np.mean(dim_z[:3]) if np.mean(dim_z[:3]) != 0 else 1e-6
        A0 = np.mean(area[:3]) if np.mean(area[:3]) != 0 else 1e-6
        Min_A0 = np.mean(min_area[:3]) if np.mean(min_area[:3]) != 0 else 1e-6
        V0 = np.mean(volume[:3]) if np.mean(volume[:3]) != 0 else 1e-6
        X0_mech = np.mean(machine_x[:3]) if np.mean(machine_x[:3]) != 0 else 1e-6

        # 3. Continuous Array Math

        # Stress calculations (kPa = mN / mm^2)
        true_stress_kpa = force_mN / area
        true_stress_max_kpa = force_mN / min_area
        eng_stress_kpa = force_mN / A0
        eng_stress_max_kpa = force_mN / Min_A0

        # Stretch Ratios (Lambda)
        stretch_x_opt = dim_x_opt / X0_opt
        stretch_y = dim_y / Y0
        stretch_z = dim_z / Z0
        stretch_x_mech = machine_x / X0_mech

        # Volumetric Ratio (J) - Incompressibility check
        volumetric_ratio = volume / V0

        # Logarithmic (True) Strain (epsilon = ln(lambda))
        true_strain_x_opt = np.log(stretch_x_opt)
        true_strain_y = np.log(stretch_y)
        true_strain_z = np.log(stretch_z)
        true_strain_x_mech = np.log(stretch_x_mech)

        # --- UPDATED POISSON'S RATIO LOGIC ---
        # Widen the deadband to 2% strain to avoid the divide-by-zero asymptote at cycle boundaries.
        # Use np.nan so the plot leaves a clean gap when resting, rather than snapping to 0.5.
        strain_threshold = 0.02

        poissons_ratio_zx = np.where(np.abs(true_strain_x_opt) > strain_threshold,
                                     -true_strain_z / true_strain_x_opt,
                                     np.nan)

        poissons_ratio_yx = np.where(np.abs(true_strain_x_opt) > strain_threshold,
                                     -true_strain_y / true_strain_x_opt,
                                     np.nan)

        # 4. Parse Cycles & Calculate Hysteresis Energy
        unique_cycles = np.unique(cycle_flags)
        cycle_parsing = {}
        energy_dissipated = []

        for c in unique_cycles:
            mask = (cycle_flags == c)
            idx = np.where(mask)[0]

            if len(idx) < 3:
                continue

            # We use the mechanical strain for the integral to prevent pixel noise
            # from causing artificial energy loops
            c_strain = true_strain_x_mech[idx]
            c_stress = true_stress_kpa[idx]
            c_dist = machine_x[idx]

            # Find the exact peak to split the cycle
            peak_relative_idx = np.argmax(c_dist)
            peak_global_idx = idx[peak_relative_idx]

            # --- Explicit Energy Integration ---
            # 1. Isolate loading and unloading arrays
            load_strain = c_strain[:peak_relative_idx + 1]
            load_stress = c_stress[:peak_relative_idx + 1]

            unload_strain = c_strain[peak_relative_idx:]
            unload_stress = c_stress[peak_relative_idx:]

            # 2. Calculate Work In (Area under loading curve)
            w_in = np.trapz(load_stress, load_strain)

            # 3. Calculate Work Out (Area under unloading curve)
            # np.trapz returns a negative value because strain is decreasing, so we use abs()
            w_out = abs(np.trapz(unload_stress, unload_strain))

            # 4. Dissipated Energy = Work In - Work Out
            dissipated = w_in - w_out

            # 5. Safeguard: Tissue cannot generate energy.
            # Any negative value is pure sensor lag/noise in the fully elastic regime.
            dissipated_mJ_mm3 = max(0.0, dissipated)

            energy_dissipated.append(dissipated_mJ_mm3)

            # Store the indices so the UI can easily color-code Loading vs. Unloading curves
            cycle_parsing[int(c)] = {
                'full_idx': idx.tolist(),
                'peak_idx': int(peak_global_idx),
                'loading_idx': idx[:peak_relative_idx + 1].tolist(),
                'unloading_idx': idx[peak_relative_idx:].tolist()
            }

        # 5. Package and Emit Payload
        mechanics_payload = {
            'time_s': time_s.tolist(),
            'true_stress_kpa': true_stress_kpa.tolist(),
            'true_stress_max_kpa': true_stress_max_kpa.tolist(),
            'eng_stress_kpa': eng_stress_kpa.tolist(),
            'eng_stress_max_kpa': eng_stress_max_kpa.tolist(),

            # Export all strain variants
            'strain_x_opt': true_strain_x_opt.tolist(),
            'strain_x_mech': true_strain_x_mech.tolist(),
            'strain_y': true_strain_y.tolist(),
            'strain_z': true_strain_z.tolist(),

            # Export all stretch variants
            'stretch_x_opt': stretch_x_opt.tolist(),
            'stretch_x_mech': stretch_x_mech.tolist(),
            'stretch_y': stretch_y.tolist(),
            'stretch_z': stretch_z.tolist(),

            'volumetric_ratio': volumetric_ratio.tolist(),
            'poissons_ratio_zx': poissons_ratio_zx.tolist(),
            'poissons_ratio_yx': poissons_ratio_yx.tolist(),
            'energy_dissipated': energy_dissipated,
            'cycle_parsing': cycle_parsing
        }

        self.mechanics_payload = mechanics_payload
        self.mechanics_available.emit(mechanics_payload)
        print("Biomechanics calculated and emitted!")

    def calculate_last_pull(self, ref_mode="cycle_start", manual_length=0.0, preload_force=0.0, cutoff_stretch=0.0):
        """
        Slices the last cycle's loading curve (up to max force) and calculates
        stretch dynamically based on the requested reference length mode.
        """
        log.info(f"Pipeline: calculate_last_pull triggered (mode={ref_mode})")

        # Failsafe: Check if geometry data exists
        if getattr(self, 'geometry_data', None) is None:
            log.warning("Pipeline: calculate_last_pull aborted -> geometry_data is None")
            return

        # Failsafe: Fallback to raw data if data_trimmed isn't available yet
        source_data = getattr(self, 'data_trimmed', None)
        if source_data is None:
            source_data = getattr(self, 'data', None)

        if source_data is None:
            log.warning("Pipeline: calculate_last_pull aborted -> both data_trimmed and data are None")
            return

        try:
            geom = self.geometry_data
            dim_x_opt = np.array(geom['dim_x'])
            area = np.array(geom['area'])

            force_mN = source_data['force']
            cycle_flags = source_data['cycle']

            # Quirk for backwards load cell data
            if "2026-02-05" in str(self.video):
                force_mN = -force_mN

        except KeyError as e:
            log.error(f"Pipeline: calculate_last_pull aborted -> Missing expected key: {e}")
            return

        # Failsafe alignment
        min_len = min(len(dim_x_opt), len(force_mN))
        if min_len == 0:
            log.warning("Pipeline: calculate_last_pull aborted -> Data arrays are empty")
            return

        dim_x_opt = dim_x_opt[:min_len]
        force_mN = force_mN[:min_len]
        cycle_flags = cycle_flags[:min_len]
        area = area[:min_len]

        # --- UPDATED: Identify the effective "Last Cycle" ---
        unique_cycles = np.unique(cycle_flags)
        if len(unique_cycles) == 0:
            log.warning("Pipeline: calculate_last_pull aborted -> No cycles found in data")
            return

        valid_cycle_found = False

        # Iterate backwards to find the first cycle with enough data points
        for cycle_candidate in reversed(unique_cycles):
            cycle_mask = (cycle_flags == cycle_candidate)
            cycle_indices = np.where(cycle_mask)[0]

            # We require at least 5 points to confidently establish a loading curve and a peak
            if len(cycle_indices) >= 5:
                last_cycle = cycle_candidate
                last_cycle_indices = cycle_indices
                valid_cycle_found = True
                log.debug(
                    f"Pipeline: Isolating effective last cycle -> Cycle {last_cycle} ({len(cycle_indices)} points)")
                break
            else:
                log.info(
                    f"Pipeline: Skipping Cycle {cycle_candidate} (only {len(cycle_indices)} points). Checking previous...")

        if not valid_cycle_found:
            log.warning("Pipeline: calculate_last_pull aborted -> No cycle with sufficient data points found.")
            return
        # ----------------------------------------------------

        # 2. Extract the raw "Last Pull" data (Start of last cycle up to max force)
        lc_force = force_mN[last_cycle_indices]
        peak_relative_idx = np.argmax(lc_force)

        pull_indices = last_cycle_indices[:peak_relative_idx + 1]
        raw_pull_dim_x = dim_x_opt[pull_indices]
        raw_pull_force = force_mN[pull_indices]
        raw_pull_area = area[pull_indices]

        # 3. Determine Reference Length (X₀) and Slicing Index
        X0 = 1.0
        slice_idx = 0  # Default to no slice (start of cycle)

        if ref_mode == "global_start":
            X0 = dim_x_opt[0]
            # slice_idx remains 0

        elif ref_mode == "cycle_start":
            X0 = raw_pull_dim_x[0]
            # slice_idx remains 0

        elif ref_mode == "manual":
            X0 = manual_length
            # If manual length is within or below the data range, slice it!
            # np.argmax returns the FIRST True value, meaning the exact point it crosses the length
            if np.any(raw_pull_dim_x >= X0):
                slice_idx = int(np.argmax(raw_pull_dim_x >= X0))
            else:
                # If manual length is vastly larger than the data, don't slice,
                # just let it plot normally (stretch will just be plotted < 1.0)
                slice_idx = 0

        elif ref_mode == "preload":
            preload_mask = raw_pull_force >= preload_force
            if np.any(preload_mask):
                slice_idx = int(np.argmax(preload_mask))
                X0 = raw_pull_dim_x[slice_idx]
                log.debug(f"Pipeline: Preload reached at local index {slice_idx}")
            else:
                log.warning(
                    f"Pipeline: Preload force {preload_force} mN never reached. Falling back to cycle start.")
                X0 = raw_pull_dim_x[0]
                slice_idx = 0

        # Prevent divide-by-zero
        if X0 <= 0:
            log.warning(f"Pipeline: Calculated X₀ was {X0}. Clamping to 1e-6 to prevent DivisionByZero.")
            X0 = 1e-6

        # 4. Slice the arrays from the determined start point to the peak
        pull_dim_x = raw_pull_dim_x[slice_idx:]
        pull_force = raw_pull_force[slice_idx:]
        pull_area = raw_pull_area[slice_idx:]

        log.debug(f"Pipeline: Final applied X₀ = {X0:.3f}, sliced {slice_idx} earlier points.")

        # 5. Calculate Responsive Metrics
        stretch = pull_dim_x / X0
        stress = pull_force / pull_area

        # --- Enforce Strictly Increasing Monotonicity ---
        if len(stretch) > 1:
            keep_mask = np.ones(len(stretch), dtype=bool)
            current_max = stretch[0]

            for i in range(1, len(stretch)):
                if stretch[i] > current_max:
                    current_max = stretch[i]
                else:
                    keep_mask[i] = False

            stretch = stretch[keep_mask]
            stress = stress[keep_mask]

            points_removed = np.sum(~keep_mask)
            if points_removed > 0:
                log.info(f"Pipeline: Monotonicity filter removed {points_removed} noisy optical points.")

        self.last_pull_stress = stress
        self.last_pull_stretch = stretch

        # --- NEW: Determine the Split Index (Cutoff) ---
        max_points = len(stretch)

        # If user hasn't set a cutoff (0.0) or it's out of bounds, default to the middle
        if cutoff_stretch <= 0.0 or not np.any(stretch >= cutoff_stretch):
            cutoff_idx = max_points // 2
            applied_cutoff = float(stretch[cutoff_idx]) if max_points > 0 else 1.0
        else:
            cutoff_idx = int(np.argmax(stretch >= cutoff_stretch))
            applied_cutoff = cutoff_stretch

        # Failsafe: Ensure the cutoff leaves at least 5 points for both sides if possible
        if cutoff_idx < 5:
            cutoff_idx = min(5, max_points // 2)
        if (max_points - cutoff_idx) < 5:
            cutoff_idx = max(max_points - 5, max_points // 2)

        log.debug(f"Pipeline: Split data at index {cutoff_idx} (λ ≈ {applied_cutoff:.3f})")

        # --- UPDATED: Iterative Linearized Stiffness Calculations ---
        # Added 'max_search_points' to strictly limit how far the algorithm can look
        def fit_linear_region(x_arr, y_arr, max_search_points, reverse=False):
            min_n = min(5, len(x_arr), max_search_points)
            search_limit = max(min_n, max_search_points)

            best_r2 = -1.0
            n_best = min_n
            m_best = 0.0
            b_best = 0.0

            if len(x_arr) >= min_n:
                for n in range(min_n, search_limit + 1):
                    s_slice = x_arr[-n:] if reverse else x_arr[:n]
                    y_slice = y_arr[-n:] if reverse else y_arr[:n]

                    slope, intercept, r_value, p_value, std_err = linregress(s_slice, y_slice)
                    r_sq = r_value ** 2

                    if r_sq > best_r2:
                        best_r2 = r_sq
                        n_best = n
                        m_best = slope
                        b_best = intercept

                lam_val = x_arr[-n_best] if reverse else x_arr[n_best - 1]
                sig_val = y_arr[-n_best] if reverse else y_arr[n_best - 1]

                return n_best, m_best, b_best, float(lam_val), float(sig_val)
            else:
                return 0, 0.0, 0.0, 0.0, 0.0

        # Execute fits using the cutoff index to define boundaries
        n0, E0, b0, lam0, sig0 = fit_linear_region(stretch, stress, max_search_points=cutoff_idx, reverse=False)
        n1, E1, b1, lam1, sig1 = fit_linear_region(stretch, stress, max_search_points=(max_points - cutoff_idx),
                                                   reverse=True)

        # Calculate Intersection (λ_m, σ_m)
        if abs(E0 - E1) > 1e-6:
            lam_m = (b1 - b0) / (E0 - E1)
            sig_m = E0 * lam_m + b0
        else:
            lam_m, sig_m = float('nan'), float('nan')

        # Emit payload to the UI (Now includes 'applied_cutoff')
        payload = {
            'stretch': stretch.tolist(),
            'stress': stress.tolist(),
            'ref_length': float(X0),
            'applied_cutoff': float(applied_cutoff),
            'stiffness': {
                'initial': {'n': n0, 'E': E0, 'b': b0, 'lambda': lam0, 'sigma': sig0},
                'terminal': {'n': n1, 'E': E1, 'b': b1, 'lambda': lam1, 'sigma': sig1},
                'intersect': {'lambda': lam_m, 'sigma': sig_m}
            }
        }

        log.info("Pipeline: calculate_last_pull emitting successfully.")
        self.last_pull_available.emit(payload)

    def calculate_relaxation(self):
        """
        Parses a stress relaxation hold, isolating the active motor ramp using mechanical strain,
        and calculating moduli using the true optical tissue strain.
        """
        print("Calculating stress relaxation metrics...")
        if getattr(self, 'mechanics_payload', None) is None:
            return

        from scipy.optimize import curve_fit

        data = self.mechanics_payload
        geom = self.geometry_data
        time_s = np.array(data['time_s'])
        stress = np.array(data['true_stress_kpa'])
        dim_z = np.array(geom['dim_z'])

        # Explicitly separate the two strain arrays for their specific jobs
        strain_mech = np.array(data['strain_x_mech'])
        strain_opt = np.array(data['strain_x_opt'])

        # 1. Isolate the Active Loading Ramp (Jaws Moving)
        # -> STRICTLY using the smooth mechanical strain as the trigger
        strain_rate = np.diff(strain_mech)

        # Use a small threshold (e.g., 5% of max speed) to ignore baseline noise
        movement_threshold = np.max(strain_rate) * 0.05
        moving_indices = np.where(strain_rate > movement_threshold)[0]

        if len(moving_indices) == 0:
            print("No jaw movement detected in data.")
            return

        # The exact frames the motor started and stopped
        ramp_start_idx = moving_indices[0]
        hold_start_idx = moving_indices[-1] + 1

        # 2. Find the True Peak Stress (Top of the compliance hump)
        peak_relative_idx = np.argmax(stress[hold_start_idx:])
        peak_stress_idx = hold_start_idx + peak_relative_idx
        peak_stress = stress[peak_stress_idx]

        # 3. Calculate Instantaneous Modulus (E_inst)
        # -> STRICTLY using the optical strain for the math
        baseline_stress = stress[ramp_start_idx]
        baseline_strain = strain_opt[ramp_start_idx]

        delta_stress = stress[hold_start_idx] - baseline_stress
        delta_strain = strain_opt[hold_start_idx] - baseline_strain

        if delta_strain > 0:
            e_inst = delta_stress / delta_strain
        else:
            e_inst = 0.0

        # 4. Prepare Hold Data for SLS Fitting
        hold_time_raw = time_s[peak_stress_idx:]
        hold_time_shifted = hold_time_raw - hold_time_raw[0]
        hold_stress = stress[peak_stress_idx:]
        final_recorded_stress = hold_stress[-1]

        def two_tau_model(t, sigma_inf, sigma_1, tau_1, sigma_2, tau_2):
            return sigma_inf + sigma_1 * np.exp(-t / tau_1) + sigma_2 * np.exp(-t / tau_2)

        total_viscous_drop = peak_stress - final_recorded_stress

        # Guesses: [Asymptote, Fast Drop, Fast Tau, Slow Drop, Slow Tau]
        p0 = [final_recorded_stress, total_viscous_drop * 0.5, 1.0, total_viscous_drop * 0.5, 50.0]
        bounds = (0, np.inf)

        try:
            popt, _ = curve_fit(two_tau_model, hold_time_shifted, hold_stress, p0=p0, bounds=bounds, maxfev=5000)

            # Unpack the raw optimized parameters
            sigma_inf_raw, s1, t1, s2, t2 = popt

            # Sort them so tau_1 is ALWAYS the "fast" phase, and tau_2 is the "slow" phase
            if t1 > t2:
                t1, t2 = t2, t1
                s1, s2 = s2, s1

            sigma_inf, sigma_1, tau_1, sigma_2, tau_2 = sigma_inf_raw, s1, t1, s2, t2

            # Generate the fitted curve array for the UI
            fitted_stress = two_tau_model(hold_time_shifted, sigma_inf, sigma_1, tau_1, sigma_2, tau_2)

        except Exception as e:
            print(f"2-Tau Curve fit failed: {e}")
            sigma_inf = final_recorded_stress
            sigma_1, tau_1 = (total_viscous_drop * 0.5), 1.0
            sigma_2, tau_2 = (total_viscous_drop * 0.5), 50.0
            fitted_stress = hold_stress

        # 6. Final Derived Metrics
        relaxed_delta_stress = sigma_inf - baseline_stress

        # -> STRICTLY using the optical strain for the math
        if delta_strain > 0:
            e_inf = relaxed_delta_stress / delta_strain
        else:
            e_inf = 0.0

        percent_relaxation = ((peak_stress - sigma_inf) / peak_stress) * 100 if peak_stress > 0 else 0.0

        # 7. Package and Emit
        relax_payload = {
            'time_s': time_s.tolist(),
            'stress_kpa': stress.tolist(),
            'dim_z': dim_z.tolist(),
            'hold_start_idx': int(hold_start_idx),
            'peak_stress_idx': int(peak_stress_idx),
            'hold_time_raw': hold_time_raw.tolist(),
            'fitted_stress': fitted_stress.tolist(),
            'metrics': {
                'peak_stress': float(peak_stress),
                'sigma_inf': float(sigma_inf),
                'sigma_1': float(sigma_1),
                'tau_1': float(tau_1),
                'sigma_2': float(sigma_2),
                'tau_2': float(tau_2),
                'e_inst': float(e_inst),
                'e_inf': float(e_inf),
                'percent_relax': float(percent_relaxation)
            }
        }

        self.relaxation_available.emit(relax_payload)
        self.relaxation_payload = relax_payload
        print("Relaxation calculated and emitted!")

    def calculate_spline(self, s_value: int) -> np.ndarray | None:
        """
        Calculates a smoothed spline for the stress-stretch data.

        Args:
            s_value: The smoothing factor 's' for the UnivariateSpline.

        Returns:
            A NumPy array of the smoothed y-values (stress) on success,
            or None if the calculation fails (e.g., insufficient data).
        """
        if getattr(self, "last_pull_stretch", None) is None or self.last_pull_stretch.size < 4:
            log.warning("Spline calculation skipped: not enough data points.")
            self.p_spline = None # Ensure old spline is cleared
            return None

        try:
            self.s_value = s_value
            spline = UnivariateSpline(self.last_pull_stretch, self.last_pull_stress, s=s_value)
            self.p_spline = spline  # Store the spline object
            return spline(self.last_pull_stretch) # Return the calculated Y-values

        except Exception as e:
            log.error(f"Error during spline calculation: {e}")
            self.p_spline = None # Ensure old spline is cleared
            return None


    ### region EXPORT TAB

    def generate_report(self):
        """Prepares the trimmed data and calculated mechanics for CSV export."""
        print("Gathering data for export...")

        if (
                self.data_trimmed is None
                or self.data_trimmed.size == 0
                or getattr(self, "geometry_data", None) is None
                or getattr(self, "mechanics_payload", None) is None
        ):
            message = "Cannot export: run Get Geometry for the active trim range first."
            print(message)
            if self.task_manager is not None:
                self.task_manager.status_updated.emit(message)
            return

        # 2. Safely handle array lengths
        num_rows = len(self.data_trimmed)

        if num_rows == 0:
            print("Cannot export: Data arrays are empty.")
            return

        # 3. Define the translation map for your headers
        header_mapping = {
            "time_s": "Time (s)",
            "force": "Force (mN)",
            "distance": "Distance (mm)",
            "cycle": "Cycle Number"
        }

        report_data = {}

        # 4. Grab columns from data_trimmed and apply mapped headers
        for raw_col in self.data_trimmed.dtype.names:
            # Use the mapped name if it exists, otherwise fallback to title-casing the raw name
            friendly_name = header_mapping.get(raw_col, raw_col.replace("_", " ").title())
            report_data[friendly_name] = self.data_trimmed[raw_col][:num_rows].tolist()

        # 5. Add the calculated mechanics columns with readable headers
        report_data["X Dimension (mm)"] = self.geometry_data["dim_x"][:num_rows]
        report_data["Y Dimension (mm)"] = self.geometry_data["dim_y"][:num_rows]
        report_data["Z Dimension (mm)"] = self.geometry_data["dim_z"][:num_rows]
        report_data["YZ Area (mm^2)"] = self.geometry_data["area"][:num_rows]
        report_data["Min YZ Area (mm^2)"] = self.geometry_data["min_area"][:num_rows]
        report_data["Volume (mm^3)"] = self.geometry_data["volume"][:num_rows]
        report_data["True Stress (kPa)"] = self.mechanics_payload["true_stress_kpa"][:num_rows]
        report_data["True Stress Max (kPa)"] = self.mechanics_payload["true_stress_max_kpa"][:num_rows]
        report_data["Stretch X Tissue"] = self.mechanics_payload["stretch_x_opt"][:num_rows]
        report_data["Stretch X Jaws"] = self.mechanics_payload["stretch_x_mech"][:num_rows]

        # --- NEW: Pad and Append Last Pull Data ---
        # Retrieve attributes safely in case calculate_last_pull hasn't been executed
        lp_stress = getattr(self, 'last_pull_stress', None)
        lp_stretch = getattr(self, 'last_pull_stretch', None)

        # Initialize columns with NaNs
        padded_lp_stress = np.full(num_rows, np.nan)
        padded_lp_stretch = np.full(num_rows, np.nan)

        if lp_stress is not None and len(lp_stress) > 0:
            fit_len = min(len(lp_stress), num_rows)
            padded_lp_stress[:fit_len] = lp_stress[:fit_len]

        if lp_stretch is not None and len(lp_stretch) > 0:
            fit_len = min(len(lp_stretch), num_rows)
            padded_lp_stretch[:fit_len] = lp_stretch[:fit_len]

        report_data["Last Pull Stress (kPa)"] = padded_lp_stress.tolist()
        report_data["Last Pull Stretch"] = padded_lp_stretch.tolist()

        # --- NEW & SAFE: Spline Tangent Modulus (Derivative) Evaluation ---
        spline_stress_targets = []
        spline_slopes = []

        p_spline = getattr(self, 'p_spline', None)

        if p_spline is not None and lp_stress is not None and lp_stretch is not None and len(lp_stress) > 0:
            max_stress = np.max(lp_stress)
            min_stress = np.min(lp_stress)

            target_val = 5.0

            if max_stress >= target_val:
                try:
                    spline_deriv = p_spline.derivative()
                except Exception as e:
                    print(f"Warning: Failed to create spline derivative: {e}")
                    spline_deriv = None

                if spline_deriv is not None:
                    while target_val <= max_stress:
                        spline_stress_targets.append(target_val)
                        slope_val = np.nan  # Default to nan for safety

                        try:
                            # Only calculate if the target stress is actually within the data bounds
                            if target_val >= min_stress:
                                idx = np.argmax(lp_stress >= target_val)

                                if idx > 0:
                                    stress_diff = lp_stress[idx] - lp_stress[idx - 1]
                                    # Prevent division by zero if consecutive stress values are identical
                                    if stress_diff > 1e-6:
                                        fraction = (target_val - lp_stress[idx - 1]) / stress_diff
                                        exact_stretch = lp_stretch[idx - 1] + fraction * (
                                                    lp_stretch[idx] - lp_stretch[idx - 1])
                                    else:
                                        exact_stretch = lp_stretch[idx]
                                else:
                                    exact_stretch = lp_stretch[0]

                                # Evaluate the derivative
                                raw_slope = float(spline_deriv(exact_stretch))

                                # Ensure it didn't return infinity or NaN
                                if np.isfinite(raw_slope):
                                    slope_val = raw_slope

                        except Exception as e:
                            # If a specific point fails, it logs silently and slope_val stays NaN
                            print(f"Warning: Derivative failed at {target_val} kPa: {e}")
                            pass

                        spline_slopes.append(slope_val)
                        target_val += 5.0

        # Pad the spline arrays to match num_rows
        padded_spline_stress = np.full(num_rows, np.nan)
        padded_spline_slopes = np.full(num_rows, np.nan)

        if len(spline_stress_targets) > 0:
            fit_len = min(len(spline_stress_targets), num_rows)
            padded_spline_stress[:fit_len] = spline_stress_targets[:fit_len]
            padded_spline_slopes[:fit_len] = spline_slopes[:fit_len]

        report_data["Tangent Modulus Eval Stress (kPa)"] = padded_spline_stress.tolist()
        report_data["Tangent Modulus (kPa)"] = padded_spline_slopes.tolist()


        if getattr(self, 'relaxation_payload', None) is not None:
            rel_data = self.relaxation_payload
            peak_idx = rel_data['peak_stress_idx']
            fit_array = rel_data['fitted_stress']
            m = rel_data['metrics']

            # A. Pad the fitted stress array with NaNs so it aligns perfectly with time
            padded_fit = np.full(num_rows, np.nan)
            available_slots = num_rows - peak_idx

            if available_slots > 0:
                fit_len = min(len(fit_array), available_slots)
                padded_fit[peak_idx: peak_idx + fit_len] = fit_array[:fit_len]

            report_data["Fitted Stress 2-Tau (kPa)"] = padded_fit.tolist()

            # B. Append scalar metrics as columns (Value in Row 1, blanks below)
            scalar_metrics = {
                "Peak Stress (kPa)": m['peak_stress'],
                "Step Modulus E_step (kPa)": m['e_inst'],
                "Relaxed Modulus E_inf (kPa)": m['e_inf'],
                "Total Relaxation (%)": m['percent_relax'],
                "Eq Stress Model (kPa)": m['sigma_inf'],
                "Fast Visc Drop (kPa)": m['sigma_1'],
                "Fast Tau (s)": m['tau_1'],
                "Slow Visc Drop (kPa)": m['sigma_2'],
                "Slow Tau (s)": m['tau_2']
            }

            for col_name, val in scalar_metrics.items():
                col_data = np.full(num_rows, np.nan)
                if num_rows > 0:
                    col_data[0] = val  # Drop the metric into the very first row
                report_data[col_name] = col_data.tolist()


        # 6. Pass to the CSV writer
        self.write_csv_report(report_data, num_rows)


    def write_csv_report(self, report_data: dict, num_rows: int):
        """Handles the logic of writing the final report to a CSV file."""
        filename = os.path.splitext(os.path.basename(self.video))[0].removesuffix("_video")
        folder = os.path.dirname(self.video)
        filepath = Path(f"{folder}/{filename}_results.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Handle existing files by incrementing a counter
        i = 1
        base_stem = filepath.stem
        while filepath.exists():
            i += 1
            filepath = filepath.with_name(f"{base_stem}_{i}.csv")

        header = ",".join(report_data.keys())
        with open(filepath, 'w', newline='') as f:
            f.write(header + '\n')
            for i in range(num_rows):
                row_values = [col[i] for col in report_data.values()]
                row_values = [np.nan if not np.isfinite(val) else val for val in row_values]
                formatted_row = [('' if np.isnan(val) else f'{val:.10g}') for val in row_values]
                f.write(','.join(formatted_row) + '\n')
        print(f"Report successfully written to {filepath}")
        self.exported_file = filepath

    # endregion


    ### region SESSION handling

    def get_state(self, debug_json=False):
        """
        Collects all serializable attributes into a dictionary for saving.

        This method handles the conversion of non-serializable objects (like QPointF)
        into a format that can be pickled.
        """
        # Start with a dictionary of all of the pipeline data
        state = vars(self).copy()

        # --- 1. Dynamic Filtering ---
        keys_to_remove = []
        for key, value in state.items():
            # Filter out PySide Signals
            if isinstance(value, SignalInstance):
                keys_to_remove.append(key)
                continue

            # Filter out Qt Internals (like __METAOBJECT__)
            if key.startswith('__') or "PyCapsule" in str(type(value)):
                keys_to_remove.append(key)
                continue

            # Filter out Scipy Splines (checking type string avoids needing to import scipy here)
            if "scipy" in str(type(value)) and "Spline" in str(type(value)):
                keys_to_remove.append(key)
                continue

        # Remove the dynamically identified keys
        for key in keys_to_remove:
            state.pop(key, None)

        # --- 2. Manual Exclusions ---
        # Explicitly remove large or runtime-only objects
        keys_to_exclude = [
            'task_manager', 'loaded_state', 'backup_state', 'geometry_in_progress',
            'voxel_arrays',
            # 'left_leveled', 'right_leveled',
            # 'left_threshed', 'right_threshed',
            # 'left_threshed_old', 'right_threshed_old',
            'objectNameChanged'  # Sometimes appears as a string/internal in Qt
        ]

        for k in keys_to_exclude:
            try:
                state.pop(k, None)
            except:
                pass

        # 3. Convert any non-serializable objects into a savable format.
        # Here, we convert the list of QPointF objects into a list of tuples.
        #if 'key_locations' in state and state['key_locations']:
            # This list comprehension creates a new list of simple (x, y) tuples
        #    state['key_locations'] = [(p.x, p.y) for p in state['key_locations']]

        trace_bad_types(state)

        state = serialize_objects(state)

        if not debug_json:
            print("\n--- Running JSON Serialization Check ---")
            for key, value in state.items():
                try:
                    # this will raise TypeError (or OverflowError) if not serializable
                    json.dumps(value)
                except (TypeError, OverflowError) as e:
                    print(f"[JSON SERIALIZATION FAILED] key='{key}':")
                    print(f"  • Type : {type(value).__name__}")
                    #print(f"  • Value: {value!r}")
                    print(f"  • Error: {e}")
            print("--- Debug Check Complete ---\n")

        log.info(f"Analysis state compiled")

        print("\n--- JSON Size Analysis (per key) ---")
        sizes = {}
        for key, value in state.items():
            try:
                j = json.dumps(value)
                size_bytes = len(j.encode('utf-8'))
                sizes[key] = size_bytes
            except Exception:
                sizes[key] = None
        # Sort descending by size (None last)
        for key, sz in sorted(sizes.items(),
                              key=lambda kv: (kv[1] is None, kv[1]),
                              reverse=True):
            if sz is None:
                print(f"{key:30s}: <failed to serialize>")
            else:
                print(f"{key:30s}: {sz/1e6:8.3f} MB")
        print("--- Size Analysis Complete ---\n")

        return state



    def load_session(self, state_dict):
        """Restore a saved analysis using the pipeline's current signal contract."""
        fixed_dict = deserialize_objects(state_dict)
        has_saved_start_time = "trim_start_time" in fixed_dict
        has_saved_end_time = "trim_end_time" in fixed_dict
        has_saved_frame_indices = "active_frame_indices" in fixed_dict
        runtime_only_keys = {"task_manager", "geometry_in_progress", "loaded_state"}
        for k, v in fixed_dict.items():
            if k not in runtime_only_keys:
                setattr(self, k, v)

        # Migrate sessions created before two-sided trimming was introduced.
        source_times = None
        if self.data is not None:
            try:
                source_times = np.asarray(self.data["time_s"], dtype=float)
            except (KeyError, TypeError, ValueError):
                source_times = None

        if source_times is not None and source_times.size > 0:
            if not has_saved_start_time:
                self.trim_start_time = float(np.min(source_times))
            if not has_saved_end_time:
                self.trim_end_time = float(getattr(self, "trim_time", np.max(source_times)))
            self.trim_time = self.trim_end_time

            if not has_saved_frame_indices:
                mask = (
                    (source_times >= self.trim_start_time)
                    & (source_times <= self.trim_end_time)
                )
                self.active_frame_indices = np.flatnonzero(mask).astype(np.int64, copy=False)

        if self.data_trimmed is not None and self.data_trimmed.size > 0:
            try:
                (
                    self.max_distance_data_index,
                    self.min_distance_data_index,
                    self.max_distance_index,
                    self.min_distance_index,
                ) = map_distance_extrema_to_source(
                    self.data_trimmed,
                    self.active_frame_indices,
                )
            except ValueError as exc:
                log.warning("Could not restore trim index mapping from saved state: %s", exc)

        self.trim_revision = int(getattr(self, "trim_revision", 0))
        self.geometry_in_progress = False
        self.loaded_state = True
        self.backup_state = fixed_dict
        self.refresh_session()

    def refresh_session(self):
        """Publish restored state through signals that exist in the current UI."""
        if self.video:
            if not Path(self.video).exists():
                mkv_path = Path(self.video).with_suffix(".mkv")
                if mkv_path.exists():
                    log.info("Found MKV in place of original video file.")
                    self.video = str(mkv_path)

        self.plot_selection_changed.emit(self.plot_selection)
        self.cycle_selection_changed.emit(self.cycle_selection)
        self.known_length_changed.emit(self.known_length)
        self.pixel_length_changed.emit(self.pixel_length)
        self.scale_is_manual_changed.emit(self.scale_is_manual)
        self.manual_conversion_factor_changed.emit(self.manual_conversion_factor)
        self.conversion_factor_changed.emit(self.conversion_factor)
        self.scale_line_coords_changed.emit(self.scale_line_coords)

        if isinstance(self.left_image, np.ndarray):
            self.left_image_changed.emit(self.left_image)

        def cached_frame(index):
            if not isinstance(self.frame_data, dict):
                return None
            return self.frame_data.get(index, self.frame_data.get(str(index)))

        min_frame = cached_frame(self.min_distance_index)
        max_frame = cached_frame(self.max_distance_index)
        if isinstance(min_frame, np.ndarray):
            self.roi_min_image_loaded.emit(min_frame)
        if isinstance(max_frame, np.ndarray):
            self.roi_max_image_loaded.emit(max_frame)

        if isinstance(self.data, dict) and self.data:
            self.buti_settings = extract_buti_settings(self.data)
            self.data_available.emit(self.data)
        if self.data_trimmed is not None and self.data_trimmed.size > 0:
            self.trim_time_changed.emit(self.trim_time)
            self.trim_range_changed.emit(self.trim_start_time, self.trim_end_time)
            self.trimmed_data_available.emit(self.data_trimmed)

        crop_pixmaps = []
        for target in ("min", "max"):
            for roi in self.roi_data.get(target, []):
                crop = roi.get("crop_img")
                if isinstance(crop, np.ndarray) and crop.size > 0:
                    crop_pixmaps.append(numpy_to_qpixmap(crop))
        if crop_pixmaps:
            self.cropped_images_ready.emit(crop_pixmaps)

        self.request_dimension_images()
        if getattr(self, "geometry_data", None) is not None:
            self.geometry_available.emit(self.geometry_data)
        if getattr(self, "mechanics_payload", None) is not None:
            self.mechanics_available.emit(self.mechanics_payload)
        if getattr(self, "relaxation_payload", None) is not None:
            self.relaxation_available.emit(self.relaxation_payload)
        if hasattr(self, "s_value"):
            self.s_value_changed.emit(self.s_value)

        self.state_loaded.emit()
        log.info("Saved analysis session restored.")


    # endregion

def trace_bad_types(obj, path="state"):
    """Recursively hunts for numpy dictionary keys and QRects, printing their exact path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Check for numpy keys (which crash json.dump)
            if hasattr(k, 'item') and type(k).__module__ == 'numpy':
                log.warning(f"Found numpy key ({type(k).__name__}) at: {path} -> {k}")

            # Recurse deeper
            trace_bad_types(v, f"{path}['{k}']")

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            trace_bad_types(v, f"{path}[{i}]")

    elif "QRect" in str(type(obj)):
        log.warning(f"Found QRect at: {path}")
