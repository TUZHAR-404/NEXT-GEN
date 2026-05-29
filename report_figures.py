"""
Auto-generate all publication-quality figures for research report.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import seaborn as sns
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ReportFigureGenerator:
    """Generate all figures needed for research report."""
    
    def __init__(
        self,
        results_dir: Path,
        data_dir: Path,
        figures_dir: Path,
        dpi: int = 300
    ):
        """
        Args:
            results_dir: Directory with training results and XAI evaluation
            data_dir: Directory with dataset
            figures_dir: Output directory for figures
            dpi: DPI for saved figures
        """
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.dpi = dpi
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def fig1_sample_images(self, class_names: List[str], sample_paths: Dict[str, str]):
        """Generate 3x3 grid of sample images (one per class)."""
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        for idx, (class_name, path) in enumerate(sample_paths.items()):
            row = idx
            for col in range(3):
                try:
                    img = Image.open(path).convert('RGB')
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(class_name, fontsize=14, fontweight='bold')
                    axes[row, col].axis('off')
                except:
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig1_sample_images.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig1_sample_images.png")
        plt.close()
    
    def fig2_training_curves(self, model_types: List[str] = ['resnet', 'vit']):
        """Generate training curves for both models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training and Validation Curves', fontsize=16, fontweight='bold', y=0.995)
        
        for idx, model_type in enumerate(model_types):
            results_path = self.results_dir / model_type / f'{model_type}_results.json'
            
            if not results_path.exists():
                logger.warning(f"Results not found: {results_path}")
                continue
            
            with open(results_path) as f:
                results = json.load(f)
            
            history = results['history']
            epochs = np.arange(1, len(history['train_loss']) + 1)
            
            # Loss
            axes[0, idx].plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2, markersize=4)
            axes[0, idx].plot(epochs, history['val_loss'], 's-', label='Val', linewidth=2, markersize=4)
            axes[0, idx].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[0, idx].set_ylabel('Loss', fontsize=12, fontweight='bold')
            axes[0, idx].set_title(f'{model_type.upper()} - Loss', fontsize=13, fontweight='bold')
            axes[0, idx].legend(fontsize=11)
            axes[0, idx].grid(True, alpha=0.3)
            
            # F1 Score
            axes[1, idx].plot(epochs, history['val_f1'], 'D-', label='F1', linewidth=2, markersize=5, color='green')
            axes[1, idx].fill_between(epochs, history['val_f1'], alpha=0.3, color='green')
            axes[1, idx].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1, idx].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
            axes[1, idx].set_title(f'{model_type.upper()} - F1 Score', fontsize=13, fontweight='bold')
            axes[1, idx].set_ylim([0, 1.05])
            axes[1, idx].legend(fontsize=11)
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig2_training_curves.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig2_training_curves.png")
        plt.close()
    
    def fig3_confusion_matrices(self, class_names: List[str], model_types: List[str] = ['resnet', 'vit']):
        """Generate side-by-side confusion matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Confusion Matrices - Test Set', fontsize=16, fontweight='bold')
        
        for idx, model_type in enumerate(model_types):
            results_path = self.results_dir / model_type / f'{model_type}_results.json'
            
            if not results_path.exists():
                logger.warning(f"Results not found: {results_path}")
                continue
            
            with open(results_path) as f:
                results = json.load(f)
            
            cm = np.array(results['test_metrics']['confusion_matrix'])
            
            im = axes[idx].imshow(cm, cmap='Blues')
            axes[idx].set_xticks(np.arange(len(class_names)))
            axes[idx].set_yticks(np.arange(len(class_names)))
            axes[idx].set_xticklabels(class_names, fontsize=11)
            axes[idx].set_yticklabels(class_names, fontsize=11)
            axes[idx].set_ylabel('True Label', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{model_type.upper()}', fontsize=13, fontweight='bold')
            
            # Add text
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = axes[idx].text(j, i, cm[i, j],
                                        ha="center", va="center",
                                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                                        fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=axes[idx], label='Count')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig3_confusion_matrices.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig3_confusion_matrices.png")
        plt.close()
    
    def fig4_gradcam_comparison(self):
        """Generate all 4 Grad-CAM variants on same image."""
        # Find a COVID sample
        sample_dir = Path('results/resnet/gradcam')
        
        if not sample_dir.exists():
            logger.warning("Grad-CAM results not found")
            return
        
        covid_files = sorted([f for f in sample_dir.glob('gradcam_COVID_*.png')])
        
        if len(covid_files) == 0:
            logger.warning("No Grad-CAM COVID samples found")
            return
        
        # Load images
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Grad-CAM Variants Comparison (COVID Sample)', fontsize=16, fontweight='bold')
        
        methods = ['gradcam_COVID_0.png', 'gradcam_pp_COVID_0.png', 'eigencam_COVID_0.png', 'scorecam_COVID_0.png']
        titles = ['GradCAM', 'GradCAM++', 'EigenCAM', 'ScoreCAM']
        
        for idx, (method, title) in enumerate(zip(methods, titles)):
            path = sample_dir / method
            if path.exists():
                img = Image.open(path)
                row = idx // 2
                col = idx % 2
                axes[row, col].imshow(img)
                axes[row, col].set_title(title, fontsize=14, fontweight='bold')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig4_gradcam_comparison.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig4_gradcam_comparison.png")
        plt.close()
    
    def fig5_resnet_vs_vit_xai(self):
        """Compare ResNet Grad-CAM vs ViT Attention Rollout."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ResNet-50 vs Vision Transformer - XAI Methods', fontsize=16, fontweight='bold')
        
        resnet_sample = Path('results/resnet/gradcam/gradcam_COVID_0.png')
        vit_sample = Path('results/vit/attention_rollout/attention_rollout_COVID_0.png')
        
        # Show original + ResNet + ViT for 3 samples
        samples = [
            ('results/resnet/gradcam/gradcam_COVID_0.png', 'results/vit/attention_rollout/attention_rollout_COVID_0.png', 'COVID'),
            ('results/resnet/gradcam/gradcam_NORMAL_0.png', 'results/vit/attention_rollout/attention_rollout_NORMAL_0.png', 'NORMAL'),
            ('results/resnet/gradcam/gradcam_Viral_Pneumonia_0.png', 'results/vit/attention_rollout/attention_rollout_Viral_Pneumonia_0.png', 'Viral Pneumonia'),
        ]
        
        for col, (resnet_path, vit_path, class_name) in enumerate(samples):
            # ResNet
            if Path(resnet_path).exists():
                img = Image.open(resnet_path)
                axes[0, col].imshow(img)
                axes[0, col].set_title(f'{class_name}\nGradCAM (ResNet)', fontsize=12, fontweight='bold')
                axes[0, col].axis('off')
            
            # ViT
            if Path(vit_path).exists():
                img = Image.open(vit_path)
                axes[1, col].imshow(img)
                axes[1, col].set_title(f'{class_name}\nAttention Rollout (ViT)', fontsize=12, fontweight='bold')
                axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig5_resnet_vs_vit_xai.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig5_resnet_vs_vit_xai.png")
        plt.close()
    
    def fig6_metrics_table(self):
        """Generate heatmap of XAI evaluation metrics."""
        metrics_path = Path('results/resnet/xai_metrics.csv')
        
        if not metrics_path.exists():
            logger.warning("XAI metrics file not found")
            return
        
        df = pd.read_csv(metrics_path)
        
        # Aggregate by method
        method_metrics = df.groupby('method')[['insertion', 'deletion', 'aopc', 'entropy']].mean()
        
        # Normalize
        method_metrics_norm = (method_metrics - method_metrics.min()) / (method_metrics.max() - method_metrics.min() + 1e-8)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(method_metrics_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(method_metrics_norm.columns)))
        ax.set_yticks(np.arange(len(method_metrics_norm.index)))
        ax.set_xticklabels(method_metrics_norm.columns, fontsize=12, fontweight='bold')
        ax.set_yticklabels(method_metrics_norm.index, fontsize=12, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add text
        for i in range(len(method_metrics_norm.index)):
            for j in range(len(method_metrics_norm.columns)):
                val = method_metrics_norm.values[i, j]
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center",
                             color="black", fontsize=11, fontweight='bold')
        
        ax.set_title('XAI Evaluation Metrics Heatmap (Normalized)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label='Score')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig6_metrics_table.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig6_metrics_table.png")
        plt.close()
    
    def fig7_aopc_curves(self):
        """Plot AOPC curves for all methods."""
        metrics_path = Path('results/resnet/xai_metrics.csv')
        
        if not metrics_path.exists():
            logger.warning("XAI metrics file not found")
            return
        
        df = pd.read_csv(metrics_path)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        methods = df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method]
            aopc_mean = method_data['aopc'].mean()
            ax.scatter([], [], s=100, c=[color], label=f'{method} (μ={aopc_mean:.4f})')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('AOPC Score', fontsize=12, fontweight='bold')
        ax.set_title('AOPC Scores Across XAI Methods', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Box plot
        plt.figure(figsize=(12, 7))
        df_pivot = df.pivot_table(values='aopc', columns='method')
        df_pivot.plot(kind='box', figsize=(12, 7))
        plt.ylabel('AOPC Score', fontsize=12, fontweight='bold')
        plt.title('AOPC Score Distribution by Method', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig7_aopc_curves.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig7_aopc_curves.png")
        plt.close()
    
    def fig8_bonus_raaopc(self):
        """Compare standard AOPC vs RA-AOPC."""
        raaopc_path = Path('results/resnet/bonus/ra_aopc_comparison.csv')
        
        if not raaopc_path.exists():
            logger.warning("RA-AOPC comparison file not found")
            return
        
        df = pd.read_csv(raaopc_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Region-Aware AOPC Analysis', fontsize=16, fontweight='bold')
        
        # Bar chart comparison
        methods = df['Method'].unique()
        x = np.arange(len(methods))
        width = 0.35
        
        standard_means = []
        ra_means = []
        
        for method in methods:
            method_data = df[df['Method'] == method]
            standard_means.append(method_data['Standard_AOPC'].mean())
            ra_means.append(method_data['RA_AOPC'].mean())
        
        axes[0].bar(x - width/2, standard_means, width, label='Standard AOPC', alpha=0.8)
        axes[0].bar(x + width/2, ra_means, width, label='RA-AOPC', alpha=0.8)
        axes[0].set_xlabel('XAI Method', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('AOPC Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Standard AOPC vs RA-AOPC', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Scatter plot: Difference
        axes[1].scatter(df['Standard_AOPC'], df['RA_AOPC'], alpha=0.6, s=80)
        lim = max(df['Standard_AOPC'].max(), df['RA_AOPC'].max())
        axes[1].plot([0, lim], [0, lim], 'k--', alpha=0.5, label='No difference')
        axes[1].set_xlabel('Standard AOPC', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('RA-AOPC', fontsize=12, fontweight='bold')
        axes[1].set_title('Score Comparison', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig8_bonus_raaopc.png', dpi=self.dpi, bbox_inches='tight')
        logger.info("Generated fig8_bonus_raaopc.png")
        plt.close()
    
    def generate_all_figures(self, class_names: List[str], sample_paths: Optional[Dict] = None):
        """Generate all figures."""
        logger.info("Generating report figures...")
        
        if sample_paths:
            self.fig1_sample_images(class_names, sample_paths)
        
        self.fig2_training_curves()
        self.fig3_confusion_matrices(class_names)
        self.fig4_gradcam_comparison()
        self.fig5_resnet_vs_vit_xai()
        self.fig6_metrics_table()
        self.fig7_aopc_curves()
        self.fig8_bonus_raaopc()
        
        logger.info(f"All figures saved to {self.figures_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='aims-dtu/Covid-19 dataset')
    parser.add_argument('--figures_dir', type=str, default='results/figures')
    parser.add_argument('--dpi', type=int, default=300)
    
    args = parser.parse_args()
    
    class_names = ['COVID', 'NORMAL', 'VIRAL_PNEUMONIA']
    
    generator = ReportFigureGenerator(
        results_dir=Path(args.results_dir),
        data_dir=Path(args.data_dir),
        figures_dir=Path(args.figures_dir),
        dpi=args.dpi
    )
    
    generator.generate_all_figures(class_names)
    logger.info("Report figure generation complete!")


if __name__ == '__main__':
    main()
