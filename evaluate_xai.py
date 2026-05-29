"""
Comprehensive XAI evaluation script: Generate saliency maps and compute metrics.
"""

import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from models.resnet_model import create_resnet_model
from models.vit_model import create_vit_model
from data.dataset import load_dataset
from explainability.gradcam import (
    GradCAM, GradCAMPlusPlus, EigenCAM, ScoreCAM, overlay_heatmap_on_image
)
from explainability.attention_rollout import AttentionRollout
from explainability.attribution_maps import IntegratedGradientsExplainer
from evaluation.metrics import (
    InsertionMetric, DeletionMetric, AOPCMetric, SaliencyEntropy, compute_all_metrics
)
from evaluation.visualize import save_saliency_overlay
from bonus.region_aware_aopc import RegionAwareAOPC, compare_aopc_methods, statistical_comparison


class XAIEvaluator:
    """Complete XAI evaluation pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: torch.device,
        class_names: List[str],
        results_dir: Path
    ):
        """
        Args:
            model: Trained PyTorch model
            model_type: 'resnet' or 'vit'
            device: Torch device
            class_names: List of class names
            results_dir: Directory to save results
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.class_names = class_names
        self.results_dir = results_dir
        self.model.eval()
        
        # Create subdirectories
        self.gradcam_dir = results_dir / 'gradcam'
        self.attention_dir = results_dir / 'attention_rollout'
        self.ig_dir = results_dir / 'integrated_gradients'
        self.bonus_dir = results_dir / 'bonus'
        
        for d in [self.gradcam_dir, self.attention_dir, self.ig_dir, self.bonus_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize explainers
        if model_type == 'resnet':
            target_layer = model.get_target_layer()
            self.gradcam = GradCAM(model, target_layer)
            self.gradcam_pp = GradCAMPlusPlus(model, target_layer)
            self.eigencam = EigenCAM(model, target_layer)
            self.scorecam = ScoreCAM(model, target_layer)
        
        if model_type == 'vit':
            self.attention_rollout = AttentionRollout(model)
        
        self.ig = IntegratedGradientsExplainer(model)
        
        # Metrics
        self.insertion = InsertionMetric(model)
        self.deletion = DeletionMetric(model)
        self.aopc = AOPCMetric(model)
        self.entropy = SaliencyEntropy()
        self.ra_aopc = RegionAwareAOPC(model)
    
    def generate_gradcam_maps(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: int,
        class_name: str,
        idx: int
    ) -> Dict[str, np.ndarray]:
        """Generate Grad-CAM variants."""
        maps = {}
        
        with torch.no_grad():
            # GradCAM
            cam_gradcam = self.gradcam(input_tensor, target_class)
            viz_gradcam = overlay_heatmap_on_image(image, cam_gradcam, alpha=0.5)
            maps['GradCAM'] = cam_gradcam
            
            # GradCAM++
            cam_pp = self.gradcam_pp(input_tensor, target_class)
            viz_pp = overlay_heatmap_on_image(image, cam_pp, alpha=0.5)
            maps['GradCAMPlusPlus'] = cam_pp
            
            # EigenCAM
            cam_eigen = self.eigencam(input_tensor, target_class)
            viz_eigen = overlay_heatmap_on_image(image, cam_eigen, alpha=0.5)
            maps['EigenCAM'] = cam_eigen
            
            # ScoreCAM
            cam_score = self.scorecam(input_tensor, target_class)
            viz_score = overlay_heatmap_on_image(image, cam_score, alpha=0.5)
            maps['ScoreCAM'] = cam_score
        
        # Save visualizations
        Image.fromarray(viz_gradcam).save(self.gradcam_dir / f"gradcam_{class_name}_{idx}.png")
        Image.fromarray(viz_pp).save(self.gradcam_dir / f"gradcam_pp_{class_name}_{idx}.png")
        Image.fromarray(viz_eigen).save(self.gradcam_dir / f"eigencam_{class_name}_{idx}.png")
        Image.fromarray(viz_score).save(self.gradcam_dir / f"scorecam_{class_name}_{idx}.png")
        
        return maps
    
    def generate_attention_rollout_maps(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: int,
        class_name: str,
        idx: int
    ) -> Dict[str, np.ndarray]:
        """Generate Attention Rollout map."""
        maps = {}
        
        with torch.no_grad():
            cam_attention = self.attention_rollout(input_tensor, target_class)
            viz_attention = overlay_heatmap_on_image(image, cam_attention, alpha=0.5)
            maps['AttentionRollout'] = cam_attention
        
        # Save visualization
        Image.fromarray(viz_attention).save(self.attention_dir / f"attention_rollout_{class_name}_{idx}.png")
        
        return maps
    
    def generate_integrated_gradients_maps(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: int,
        class_name: str,
        idx: int
    ) -> Dict[str, np.ndarray]:
        """Generate Integrated Gradients map."""
        maps = {}
        
        with torch.no_grad():
            cam_ig = self.ig(input_tensor, target_class, n_steps=50)
            viz_ig = overlay_heatmap_on_image(image, cam_ig, alpha=0.5, colormap_name='hot')
            maps['IntegratedGradients'] = cam_ig
        
        # Save visualization
        Image.fromarray(viz_ig).save(self.ig_dir / f"integrated_gradients_{class_name}_{idx}.png")
        
        return maps
    
    def evaluate_test_set(
        self,
        test_loader,
        test_paths: List[str],
        test_labels: List[int],
        max_images_per_class: int = 7
    ):
        """Evaluate XAI on test set."""
        
        # Select balanced subset
        selected_indices = []
        class_counts = {i: 0 for i in range(len(self.class_names))}
        
        for idx, label in enumerate(test_labels):
            if class_counts[label] < max_images_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
        
        logger.info(f"Evaluating on {len(selected_indices)} test images")
        
        # Results storage
        all_metrics = {method: [] for method in ['GradCAM', 'GradCAMPlusPlus', 'EigenCAM', 
                                                   'ScoreCAM', 'AttentionRollout', 'IntegratedGradients']}
        aopc_vs_raaopc = []
        
        for eval_idx, img_idx in enumerate(tqdm(selected_indices, desc="Evaluating XAI")):
            
            # Load image
            img_path = test_paths[img_idx]
            true_label = test_labels[img_idx]
            
            image_pil = Image.open(img_path).convert('RGB')
            image_np = np.array(image_pil) / 255.0
            
            # Preprocess for model
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_label = output.argmax(dim=1).item()
                pred_prob = F.softmax(output, dim=1)[0, pred_label].item()
            
            class_name = self.class_names[true_label]
            
            # Generate saliency maps
            saliency_maps = {}
            
            if self.model_type == 'resnet':
                saliency_maps.update(self.generate_gradcam_maps(image_np, input_tensor, pred_label, class_name, eval_idx))
            
            if self.model_type == 'vit':
                saliency_maps.update(self.generate_attention_rollout_maps(image_np, input_tensor, pred_label, class_name, eval_idx))
            
            saliency_maps.update(self.generate_integrated_gradients_maps(image_np, input_tensor, pred_label, class_name, eval_idx))
            
            # Compute metrics for each saliency map
            for method_name, sal_map in saliency_maps.items():
                try:
                    metrics = compute_all_metrics(self.model, image_np, sal_map, pred_label)
                    metrics['image_idx'] = img_idx
                    metrics['method'] = method_name
                    metrics['true_label'] = true_label
                    metrics['pred_label'] = pred_label
                    metrics['pred_prob'] = pred_prob
                    metrics['class_name'] = class_name
                    
                    if method_name in all_metrics:
                        all_metrics[method_name].append(metrics)
                except Exception as e:
                    logger.warning(f"Error computing metrics for {method_name}: {e}")
            
            # RA-AOPC comparison
            if 'GradCAM' in saliency_maps:
                try:
                    comp = self.ra_aopc.compare_with_standard_aopc(image_np, saliency_maps['GradCAM'], pred_label)
                    comp['image_idx'] = img_idx
                    comp['class_name'] = class_name
                    aopc_vs_raaopc.append(comp)
                except Exception as e:
                    logger.warning(f"Error in RA-AOPC: {e}")
        
        # Save metrics results
        metrics_df_list = []
        for method, metrics_list in all_metrics.items():
            if metrics_list:
                method_df = pd.DataFrame(metrics_list)
                metrics_df_list.append(method_df)
        
        if metrics_df_list:
            metrics_df = pd.concat(metrics_df_list, ignore_index=True)
            metrics_df.to_csv(self.results_dir / 'xai_metrics.csv', index=False)
            logger.info(f"Saved XAI metrics to {self.results_dir / 'xai_metrics.csv'}")
        
        # Save RA-AOPC results
        if aopc_vs_raaopc:
            raaopc_df = pd.DataFrame(aopc_vs_raaopc)
            raaopc_df.to_csv(self.bonus_dir / 'ra_aopc_comparison.csv', index=False)
            logger.info(f"Saved RA-AOPC comparison to {self.bonus_dir / 'ra_aopc_comparison.csv'}")
            
            # Statistical test
            stats = statistical_comparison(
                raaopc_df['standard_aopc'].values,
                raaopc_df['ra_aopc'].values
            )
            with open(self.bonus_dir / 'ra_aopc_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"RA-AOPC t-test p-value: {stats['p_value']:.4f}")
        
        return metrics_df if metrics_df_list else None


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet', 'vit', 'both'], default='both')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--data_root', type=str, default='aims-dtu/Covid-19 dataset')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_loader, val_loader, test_loader, metadata = load_dataset(
        data_root=Path(args.data_root),
        batch_size=1,
        seed=42
    )
    
    class_names = metadata['class_names']
    test_paths = metadata['test_paths']
    test_labels = metadata['test_labels']
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_eval = ['resnet', 'vit'] if args.model == 'both' else [args.model]
    
    for model_type in models_to_eval:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {model_type.upper()}")
        logger.info(f"{'='*80}\n")
        
        # Load model
        checkpoint_path = Path(args.checkpoint_dir) / f"{model_type}_best.pt"
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        if model_type == 'resnet':
            model = create_resnet_model(num_classes=3, pretrained=True)
        else:
            model = create_vit_model(num_classes=3, pretrained=True)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        evaluator = XAIEvaluator(model, model_type, device, class_names, results_dir / model_type)
        metrics_df = evaluator.evaluate_test_set(test_loader, test_paths, test_labels)
        
        logger.info(f"\nCompleted evaluation for {model_type}")
    
    logger.info("\nXAI evaluation complete!")


if __name__ == '__main__':
    main()
