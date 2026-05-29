"""
XAI evaluation metrics: Insertion, Deletion, AOPC, Entropy.
"""

import logging
from typing import Tuple, List, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InsertionMetric:
    """
    Insertion metric: Start with blurred image, progressively insert top-k% pixels
    ranked by saliency. Higher score indicates better saliency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        blur_sigma: float = 10.0,
        num_steps: int = 20
    ):
        """
        Args:
            model: PyTorch model
            blur_sigma: Gaussian blur sigma
            num_steps: Number of perturbation steps
        """
        self.model = model
        self.blur_sigma = blur_sigma
        self.num_steps = num_steps
        self.device = next(model.parameters()).device
    
    def __call__(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        target_class: int
    ) -> float:
        """
        Compute insertion metric.
        
        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            saliency_map: Saliency map (H, W) in [0, 1]
            target_class: Target class index
            
        Returns:
            Insertion AUC score
        """
        # Blur the image
        blurred = np.stack([
            gaussian_filter(image[:, :, c], sigma=self.blur_sigma)
            for c in range(3)
        ], axis=2)
        
        # Normalize saliency
        saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # Get ranking
        flat_saliency = saliency_norm.flatten()
        ranking = np.argsort(-flat_saliency)  # Descending
        
        # Get number of pixels
        num_pixels = image.shape[0] * image.shape[1]
        
        # Compute scores at each step
        scores = []
        for step in range(self.num_steps + 1):
            # Number of pixels to insert
            num_insert = int((step / self.num_steps) * num_pixels)
            
            if num_insert == 0:
                current_image = blurred.copy()
            else:
                current_image = blurred.copy()
                # Unflatten indices
                indices = np.unravel_index(ranking[:num_insert], image.shape[:2])
                current_image[indices] = image[indices]
            
            # Get model score
            with torch.no_grad():
                img_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                output = self.model(img_tensor)
                score = F.softmax(output, dim=1)[0, target_class].item()
            
            scores.append(score)
        
        # Compute AUC
        auc = np.trapz(scores, dx=1.0 / self.num_steps)
        return auc


class DeletionMetric:
    """
    Deletion metric: Start with original image, progressively delete top-k% pixels
    by saliency. Lower score indicates better saliency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        blur_sigma: float = 10.0,
        num_steps: int = 20
    ):
        """
        Args:
            model: PyTorch model
            blur_sigma: Gaussian blur sigma for deletion
            num_steps: Number of perturbation steps
        """
        self.model = model
        self.blur_sigma = blur_sigma
        self.num_steps = num_steps
        self.device = next(model.parameters()).device
    
    def __call__(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        target_class: int
    ) -> float:
        """
        Compute deletion metric.
        
        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            saliency_map: Saliency map (H, W) in [0, 1]
            target_class: Target class index
            
        Returns:
            Deletion AUC score (inverted)
        """
        # Blur for deletion regions
        blurred = np.stack([
            gaussian_filter(image[:, :, c], sigma=self.blur_sigma)
            for c in range(3)
        ], axis=2)
        
        # Normalize saliency
        saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # Get ranking
        flat_saliency = saliency_norm.flatten()
        ranking = np.argsort(-flat_saliency)  # Descending
        
        # Get number of pixels
        num_pixels = image.shape[0] * image.shape[1]
        
        # Compute scores at each step
        scores = []
        for step in range(self.num_steps + 1):
            # Number of pixels to delete
            num_delete = int((step / self.num_steps) * num_pixels)
            
            current_image = image.copy()
            if num_delete > 0:
                # Unflatten indices
                indices = np.unravel_index(ranking[:num_delete], image.shape[:2])
                current_image[indices] = blurred[indices]
            
            # Get model score
            with torch.no_grad():
                img_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                output = self.model(img_tensor)
                score = F.softmax(output, dim=1)[0, target_class].item()
            
            scores.append(score)
        
        # Compute AUC (inverted: lower is better for deletion)
        auc = np.trapz(scores, dx=1.0 / self.num_steps)
        return -auc


class AOPCMetric:
    """
    Area Over Perturbation Curve: Perturb top-k% salient pixels and measure drop in prediction.
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch model
        """
        self.model = model
        self.device = next(model.parameters()).device
    
    def __call__(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        target_class: int,
        perturbation_percentages: Optional[List[float]] = None
    ) -> Tuple[float, List[float]]:
        """
        Compute AOPC metric.
        
        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            saliency_map: Saliency map (H, W) in [0, 1]
            target_class: Target class index
            perturbation_percentages: List of perturbation percentages to test
            
        Returns:
            AOPC score, list of scores at each step
        """
        if perturbation_percentages is None:
            perturbation_percentages = [1, 5, 10, 20, 30, 50]
        
        # Normalize saliency
        saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # Get baseline score
        with torch.no_grad():
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            output = self.model(img_tensor)
            baseline_score = F.softmax(output, dim=1)[0, target_class].item()
        
        # Get ranking
        flat_saliency = saliency_norm.flatten()
        ranking = np.argsort(-flat_saliency)
        num_pixels = image.shape[0] * image.shape[1]
        
        # Compute mean pixel value for perturbation
        mean_pixel = image.mean(axis=(0, 1))
        
        scores = []
        for perc in perturbation_percentages:
            num_perturb = max(1, int((perc / 100.0) * num_pixels))
            
            current_image = image.copy()
            indices = np.unravel_index(ranking[:num_perturb], image.shape[:2])
            current_image[indices] = mean_pixel
            
            with torch.no_grad():
                img_tensor = torch.from_numpy(current_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                output = self.model(img_tensor)
                score = F.softmax(output, dim=1)[0, target_class].item()
            
            scores.append(baseline_score - score)
        
        # Compute AOPC
        aopc = np.mean(scores)
        return aopc, scores


class SaliencyEntropy:
    """
    Entropy of saliency map: Measures focus of explanation.
    """
    
    @staticmethod
    def __call__(saliency_map: np.ndarray) -> float:
        """
        Compute entropy of saliency map.
        
        Args:
            saliency_map: Saliency map (H, W) in [0, 1]
            
        Returns:
            Shannon entropy
        """
        # Normalize to probability distribution
        flat_saliency = saliency_map.flatten()
        prob = (flat_saliency - flat_saliency.min()) / (flat_saliency.max() - flat_saliency.min() + 1e-8)
        prob = prob / prob.sum()
        
        # Remove zero probabilities to avoid log(0)
        prob = prob[prob > 0]
        
        # Shannon entropy
        ent = scipy_entropy(prob, base=2)
        return float(ent)


def compute_all_metrics(
    model: nn.Module,
    image: np.ndarray,
    saliency_map: np.ndarray,
    target_class: int
) -> dict:
    """
    Compute all XAI evaluation metrics.
    
    Args:
        model: PyTorch model
        image: Input image (H, W, 3) normalized to [0, 1]
        saliency_map: Saliency map (H, W) in [0, 1]
        target_class: Target class index
        
    Returns:
        Dictionary with metric scores
    """
    metrics = {}
    
    # Insertion
    insertion = InsertionMetric(model)
    metrics['insertion'] = insertion(image, saliency_map, target_class)
    
    # Deletion
    deletion = DeletionMetric(model)
    metrics['deletion'] = deletion(image, saliency_map, target_class)
    
    # AOPC
    aopc = AOPCMetric(model)
    aopc_score, _ = aopc(image, saliency_map, target_class)
    metrics['aopc'] = aopc_score
    
    # Entropy
    entropy = SaliencyEntropy()
    metrics['entropy'] = entropy(saliency_map)
    
    return metrics
