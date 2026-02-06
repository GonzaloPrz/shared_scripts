from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlations(predictions,save_path,r,p,title=None):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x='y_pred', y='y_true', data=predictions,
        scatter_kws={'alpha': 0.6, 's': 50, 'color': '#c9a400'},  # color base
        line_kws={'color': 'black', 'linewidth': 2}
    )

    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')

    plt.text(0.05, 0.95,
            f'$r$ = {r:.2f}\n$p$ = {np.round(p,3) if p > .001 else "< .001"}',
            fontsize=20,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title(title, fontsize=25, pad=15)

    plt.tight_layout()
    plt.grid(False)

    plt.savefig(save_path, dpi=300)

    plt.close()
