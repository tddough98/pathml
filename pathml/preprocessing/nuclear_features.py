"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import cv2
import numpy as np
import pathml.core
import pathml.core.slide_data
import scipy
import scipy.stats
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.morphology import disk

from .transforms import Transform


def nuclear_statistics(mask, image):
    """
    Args:
        mask (np.ndarray): nuclear segmentation instance mask
        image (np.ndarray): grayscale image

    Returns:
        tuple of
            * mean intensity of nuclei
            * difference between mean nuclei intensity and mean background intensity
            * variance of nuclear pixels
            * skew of nuclear pixels
    """
    nuclei = np.array(image[mask > 0])
    background = np.array(image[mask == 0])
    mean_nuclei_intensity = nuclei.sum() / (
        np.size(nuclei) + 1.0e-8
    )  # avoids divide by zero
    mean_background_intensity = background.sum() / (
        np.size(background) + 1.0e-8
    )  # avoids divide by zero
    difference = abs(mean_nuclei_intensity - mean_background_intensity)
    variance = np.var(nuclei)
    skew = scipy.stats.skew(nuclei)
    return mean_nuclei_intensity, difference, variance, skew


def nuclear_glcm_statistics(mask, image):
    """
    Gray-Level Co-Occurrence Matrix (GLCM)

    Args:
        mask (np.ndarray): nuclear segmentation instance mask
        image (np.ndarray): grayscale image

    Returns:
        tuple of GLCM statistics, namely
            * contrast,
            * dissimilarity,
            * homogeneity,
            * energy, and
            * angular second moment
    """
    # Calculate the GLCM "one pixel to the right"
    glcm = graycomatrix(image * mask, [1], [0])
    filt_glcm = glcm[1:, 1:, :, :]  # Filter out the first row and column
    contrast = graycoprops(filt_glcm, prop="contrast")[0, 0]
    dissimilarity = graycoprops(filt_glcm, prop="dissimilarity")[0, 0]
    homogeneity = graycoprops(filt_glcm, prop="homogeneity")[0, 0]
    energy = graycoprops(filt_glcm, prop="energy")[0, 0]
    angular_second_moment = graycoprops(filt_glcm, prop="ASM")[0, 0]

    return contrast, dissimilarity, homogeneity, energy, angular_second_moment


class NuclearFeatures(Transform):
    """
    Summarizes an nuclear instance segmentation mask into a set of morphological features, including
        * numb
        er of nuclei
        * total nuclear area
        * number of nuclei of each type
        * total nuclear area of each cell type
        * the mean and standard deviation across all nuclei of (features from CGC-Net)
            mean nuclear intensity
            difference in intensity between nucleus and background in bounding box
            variance nuclear intensity
            skew nuclar intensity
            mean entropy in nucleus
            glcm_dissimilarity
            glcm_homogeneity
            glcm_energy
            glcm_angular_second_moment
            for ellipse approximating nuclear shape:
                eccentricity
                area
                major_axis_length
                minor_axis_length
            perimeter
            solidity
            orientation

        Args:
            n_classes (int): number of nuclear classes in tile.labels['nuclei_classes']
            name (str): name of the field to store the nuclear features in tile.labels
    """

    def __init__(self, n_classes, name):
        self.n_classes = n_classes
        self.name = name

        self.nuclear_feature_names = [
            "mean_nuclei_intensity",
            "difference",
            "variance",
            "skew",
            "mean_entropy",
            "dissimilarity",
            "homogeneity",
            "energy",
            "angular_second_moment",
            "eccentricity",
            "area",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
            "solidity",
            "orientation",
        ]

        self.feature_names = (
            ["count"]
            + [f"count_class_{c}" for c in range(self.n_classes)]
            + ["area"]
            + [f"area_class_{c}" for c in range(self.n_classes)]
            + [f"mean_{f}" for f in self.nuclear_feature_names]
            + [f"std_dev_{f}" for f in self.nuclear_feature_names]
        )

    def count(self, mask, nuclei_class_labels=None, nuclei_class=None):
        """
        Args:
            mask (np.ndarray): nuclear instance segmentation mask
            nuclei_class_labels (np.ndarray): (cell, class) pairs
            nuclei_class (str): class to count

        Returns:
            number of cells in the given class or
            total cells if no class or class labels are given
        """
        if nuclei_class_labels is None or nuclei_class is None:
            cells = np.unique(mask)
            # subtract background if present
            return len(cells) - 1 if cells[0] == 0 else len(cells)
        assert nuclei_class_labels.size == 0 or (
            nuclei_class_labels.ndim == 2 and nuclei_class_labels.shape[1] == 2
        ), "nuclei_class_labels must be (cell, class) pairs"
        # Count cells directly from the class labels (not from the segmentation mask)
        if nuclei_class_labels.size == 0:
            return 0
        class_cells = nuclei_class_labels[nuclei_class_labels[:, 1] == nuclei_class, 0]
        return len(class_cells)

    def area(self, mask, nuclei_class_labels=None, nuclei_class=None):
        """
        Args:
            mask (np.ndarray): nuclear instance segmentation mask
            nuclei_class_labels (np.ndarray): (cell, class) pairs
            nuclei_class (str): class to find area

        Returns:
            number of pixels in the given class or
            total nuclear pixels if no class or class labels are given
        """
        if nuclei_class_labels is None or nuclei_class is None:
            return np.sum(mask > 0)
        assert nuclei_class_labels.size == 0 or (
            nuclei_class_labels.ndim == 2 and nuclei_class_labels.shape[1] == 2
        ), "nuclei_class_labels must be (cell, class) pairs"
        class_mask = np.zeros(mask.shape, dtype=np.dtype("bool"))
        for n, c in nuclei_class_labels:
            class_mask[mask == n] = c == nuclei_class
        return np.sum(class_mask > 0)

    def nuclear_features(self, image, mask):
        """
        Returns an array with 16 morphological features for each nucleus in the mask.
        Adapted from https://github.com/SIAAAAAA/CGC-Net.

        Args:
            image (np.ndarray): grayscale image
            mask (np.ndarray): nuclear instance segmentation mask

        Returns:
            ndarray of 16 features for each nucleus in the mask
            rows are nuclei, each column is a feature
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        entropy = Entropy(grayscale_image, disk(3))
        binary_mask = mask > 0
        # nuclear_features = []
        nuclear_features = {}
        # nuclear_coordinates = []
        # Compute features for each nucleus in mask
        for prop in regionprops(mask):
            l, t, r, b = prop.bbox
            # Slice image, mask, and entropy images around nucleus
            nucleus_image = grayscale_image[l : r + 1, t : b + 1]
            nucleus_mask = binary_mask[l : r + 1, t : b + 1].astype(np.uint8)
            nucleus_entropy = entropy[l : r + 1, t : b + 1]
            mean_nuclei_intensity, difference, variance, skew = nuclear_statistics(
                nucleus_mask, nucleus_image
            )
            (
                _,
                dissimilarity,
                homogeneity,
                energy,
                angular_second_moment,
            ) = nuclear_glcm_statistics(nucleus_mask, nucleus_image)
            mean_entropy = cv2.mean(nucleus_entropy, mask=nucleus_mask)[0]
            contours = cv2.findContours(
                nucleus_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )[0][0]
            area = cv2.contourArea(contours)
            # maximum avoids divide by zero in solidity
            hull_area = max(1, cv2.contourArea(contours))
            solidity = float(area) / hull_area
            if len(contours) > 4:
                _, axes, orientation = cv2.fitEllipse(contours)
                major_axis_length = max(axes)
                minor_axis_length = min(axes)
            else:
                orientation = 0
                major_axis_length = 1
                minor_axis_length = 1
            perimeter = cv2.arcLength(contours, True)
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            nuclear_features[prop.label] = np.array(
                [
                    mean_nuclei_intensity,
                    difference,
                    variance,
                    skew,
                    mean_entropy,
                    dissimilarity,
                    homogeneity,
                    energy,
                    angular_second_moment,
                    eccentricity,
                    area,
                    major_axis_length,
                    minor_axis_length,
                    perimeter,
                    solidity,
                    orientation,
                ]
            )
            # nuclear_features.append(features)
            # nuclear_coordinates.append(prop.centroid)
        return nuclear_features  # np.array(nuclear_features)

    def F(self, image, mask, nuclei_class_labels, summarize=False):
        """
        Args:
            image (np.ndarray): RGB H&E image
            mask (np.ndarray): nuclear instance segmentation
            nuclei_class_labels (np.ndarray): (cell, class) pairs
            summarize (bool): whether to average across all nuclei in the tile

        Returns:
            array of features summarizing nuclear segmentation
        """
        assert (
            mask.ndim == 3 and mask.shape[0] == 1
        ), "Nuclei segmentation instance mask must have a single channel in the first axis"
        assert image.ndim == 3 and image.shape[-1] == 3, "Image must be an RGB image"
        assert (
            mask.shape[1:] == image.shape[:2]
        ), "Image and mask must have the same shape"
        assert (
            nuclei_class_labels.ndim == 2 and nuclei_class_labels.shape[1] == 2
        ) or nuclei_class_labels.size == 0, (
            "nuclei_class_labels must be (cell, class) pairs"
        )

        mask = mask[0]

        nuclear_features = self.nuclear_features(image, mask)
        if summarize:
            nuclear_features = np.array(list(nuclear_features.values()))
            counts = [
                self.count(mask, nuclei_class_labels, i) for i in range(self.n_classes)
            ]
            areas = [
                self.area(mask, nuclei_class_labels, i) for i in range(self.n_classes)
            ]
            mean = np.mean(nuclear_features, axis=0)
            std = np.std(nuclear_features, axis=0)
            summary = np.array(
                [self.count(mask), *counts, self.area(mask), *areas, *mean, *std]
            )
            return summary
        return nuclear_features

    def apply(self, tile):
        assert isinstance(
            tile, pathml.core.tile.Tile
        ), f"tile is type {type(tile)} but must be pathml.core.tile.Tile"
        assert (
            tile.masks is not None and "nuclei" in tile.masks
        ), f"tile must have nuclear segmentation mask named 'nuclei'"
        assert (
            tile.slide_type.stain == "HE"
        ), f"Tile has slide_type.stain={tile.slide_type.stain}, but must be 'HE'"
        assert (
            "nuclei_classes" in tile.labels
        ), f"Tile must have 'nuclei_classes' labels"

        features = self.F(
            tile.image, tile.masks["nuclei"], tile.labels["nuclei_classes"]
        )
        tile.labels[self.name] = features

        return tile
