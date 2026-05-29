"""
Region-Aware AOPC: Anatomically-aware perturbation for medical imaging.

Problem: Standard AOPC perturbs individual pixels independently, ignoring spatial
correlations in medical images (lung boundaries, rib structures).

Solution: Use SLIC superpixels to segment image into coherent regions, then perturb
entire regions instead of individual pixels.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from skimage.segmentation import slic
from scipy.ndimage import gaussian_filter
import pandas as pd

logger = logging.getLogger(__name__)


class RegionAwareAOPC:
    """
    Region-Aware AOPC metric using SLIC superpixels.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_segments: int = 50,
        compactness: float = 10
    ):
        """
        Args:
            model: PyTorch model
            n_segments: Number of SLIC superpixels
            compactness: SLIC compactness parameter
        """
        self.model = model
        self.n_segments = n_segments
        self.compactness = compactness
        self.device = next(model.parameters()).device
    
    def get_superpixels(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image into SLIC superpixels.
        
        Args:
            image: Input image (H, W, 3) in [0, 1] or [0, 255]
            
        Returns:
            Superpixel labels (H, W)
        """
        # Normalize to [0, 255] for SLIC
        if image.max() <= 1:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Compute SLIC
        segments = slic(
            image_uint8,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0
        )
        
        return segments
    
    def __call__(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        target_class: int,
        perturbation_percentages: Optional[List[float]] = None
    ) -> Tuple[float, List[float], np.ndarray]:
        """
        Compute Region-Aware AOPC.
        
        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            saliency_map: Saliency map (H, W) in [0, 1]
            target_class: Target class index
            perturbation_percentages: List of perturbation percentages
            
        Returns:
            RA-AOPC score, scores at each step, superpixel labels
        """
        if perturbation_percentages is None:
            perturbation_percentages = [1, 5, 10, 20, 30, 50]
        
        # Get superpixels
        segments = self.get_superpixels(image)
        num_regions = segments.max() + 1
        
        # Compute region-wise saliency
        region_saliency = np.zeros(num_regions)
        region_counts = np.zeros(num_regions)
        
        for region_id in range(num_regions):
            mask = segments == region_id
            region_saliency[region_id] = saliency_map[mask].mean()
            region_counts[region_id] = mask.sum()
        
        # Rank regions by saliency
        region_ranking = np.argsort(-region_saliency)
        
        # Get baseline score
        with torch.no_grad():
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            output = self.model(img_tensor)
            baseline_score = F.softmax(output, dim=1)[0, target_class].item()
        
        # Compute mean pixel value for perturbation
        mean_pixel = image.mean(axis=(0, 1))
        
        scores = []
        for perc in perturbation_percentages:
            num_regions_perturb = max(1, int((perc / 100.0) * num_regions))
            
            current_image = image.copy()
            
            # Perturb regions
            for i in range(num_regions_perturb):
                region_id = region_ranking[i]
                mask = segments == region_id
                current_image[mask] = mean_pixel
            
            # Get model score
            with torch.no_grad():
                img_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                output = self.model(img_tensor)
                score = F.softmax(output, dim=1)[0, target_class].item()
            
            scores.append(baseline_score - score)
        
        # Compute RA-AOPC
        ra_aopc = np.mean(scores)
        
        return ra_aopc, scores, segments
    
    def compare_with_standard_aopc(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        target_class: int
    ) -> dict:
        """
        Compare RA-AOPC with standard AOPC.
        
        Args:
            image: Input image
            saliency_map: Saliency map
            target_class: Target class
            
        Returns:
            Dictionary with comparison metrics
        """
        from evaluation.metrics import AOPCMetric
        
        # Standard AOPC
        standard_aopc = AOPCMetric(self.model)
        aopc_score, aopc_scores = standard_aopc(image, saliency_map, target_class)
        
        # Region-Aware AOPC
        ra_aopc_score, ra_aopc_scores, segments = self(image, saliency_map, target_class)
        
        result = {
            'standard_aopc': aopc_score,
            'ra_aopc': ra_aopc_score,
            'difference': ra_aopc_score - aopc_score,
            'standard_aopc_scores': aopc_scores,
            'ra_aopc_scores': ra_aopc_scores,
            'segments': segments
        }
        
        return result


def compare_aopc_methods(
    model: nn.Module,
    images: List[np.ndarray],
    saliency_maps: List[np.ndarray],
    target_classes: List[int],
    method_names: List[str],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare standard AOPC and RA-AOPC across multiple images and methods.
    
    Args:
        model: PyTorch model
        images: List of images
        saliency_maps: List of saliency maps
        target_classes: List of target classes
        method_names: List of XAI method names used to generate saliency maps
        save_path: Path to save comparison results CSV
        
    Returns:
        DataFrame with comparison results
    """
    ra_aopc = RegionAwareAOPC(model)
    
    results = []
    
    for idx, (image, sal_map, target_class, method) in enumerate(
        zip(images, saliency_maps, target_classes, method_names)
    ):
        comparison = ra_aopc.compare_with_standard_aopc(image, sal_map, target_class)
        
        results.append({
            'Image_Index': idx,
            'Method': method,
            'Standard_AOPC': comparison['standard_aopc'],
            'RA_AOPC': comparison['ra_aopc'],
            'Difference': comparison['difference'],
            'Target_Class': target_class
        })
    
    df = pd.DataFrame(results)
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved RA-AOPC comparison to {save_path}")
    
    return df


def statistical_comparison(
    standard_aopc_scores: np.ndarray,
    ra_aopc_scores: np.ndarray
) -> dict:
    """
    Perform paired t-test between standard AOPC and RA-AOPC.
    
    Args:
        standard_aopc_scores: Array of standard AOPC scores
        ra_aopc_scores: Array of RA-AOPC scores
        
    Returns:
        Dictionary with t-test results
    """
    from scipy import stats
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(standard_aopc_scores, ra_aopc_scores)
    
    # Mean and std
    standard_mean = standard_aopc_scores.mean()
    standard_std = standard_aopc_scores.std()
    ra_mean = ra_aopc_scores.mean()
    ra_std = ra_aopc_scores.std()
    
    # Effect size (Cohen's d)
    diff = standard_aopc_scores - ra_aopc_scores
    cohens_d = diff.mean() / diff.std()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'standard_mean': standard_mean,
        'standard_std': standard_std,
        'ra_mean': ra_mean,
        'ra_std': ra_std,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
