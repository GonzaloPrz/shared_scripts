import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import json, itertools

sns.set_theme(style="white")  # white background without grid

plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize":14
    })

results_dir = Path(Path.home(),'results') if 'Users/gp' in str(Path.home()) else Path('D','CNC_Audio','gonza','results')

# Load configuration from the same folder as this script
with open(Path(__file__).parent / 'config.json', 'r') as f:
    config = json.load(f)

project_name = config['project_name']

project_name = config["project_name"]
scoring = 'roc_auc'
kfold_folder = config["kfold_folder"]
stat_folder = config["stat_folder"]
bootstrap_method = config["bootstrap_method"]
hyp_opt = bool(config["n_iter"] > 0)
try:
    feature_selection = bool(config["feature_selection"])
except:
    feature_selection = config["n_iter_features"] > 0

n_boot = int(config["n_boot"])
bayes = bool(config["bayes"])

#tasks = config['tasks']
tasks = ['Animales__P']
#dimensions = config['single_dimensions']
dimensions = ['properties']

y_labels = config['y_labels']

def _path_no_empty(*parts):
    """Join path parts ignoring empty parts."""
    parts = [p for p in parts if p not in (None, '', [])]
    return Path(*parts)


def load_feature_importance(results_dir: Path, project_name: str, task: str, dimension: str, y_label: str,
                            stat_folder: str, scoring: str, bootstrap_method: str,
                            hyp_opt: bool = False, feature_selection: bool = False,
                            filename: str = 'feature_importance_lr_top10.csv') -> pd.DataFrame:
    """Load feature importance CSV into a DataFrame. Builds path safely.

    Parameters
    - results_dir, project_name, task, dimension, y_label, stat_folder, scoring, bootstrap_method: strings/paths
    - hyp_opt, feature_selection: add these folders if True
    - filename: CSV filename
    """
    folder_parts = [results_dir, project_name, 'feature_importance_bayes' if config['bayes'] else 'feature_importance',task, dimension, y_label, scoring,stat_folder,bootstrap_method]
    if hyp_opt:
        folder_parts.append('hyp_opt')
    if feature_selection:
        folder_parts.append('feature_selection')
    folder = _path_no_empty(*folder_parts)
    file_path = folder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {file_path}")
    return pd.read_csv(file_path)


def bar_plot_feature_importance(feature_importance, feature_col='Feature', importance_col='Importance', hue_col = 'Task',
                                top_n: int = 10, sort_desc: bool = True, horizontal: bool = True,
                                figsize=(10, 6), cmap: str = None, title: str = None,
                                xlabel: str = 'Importance', ylabel: str = 'Feature',
                                ax=None, save_path: str = None, show: bool = True, normalize: bool = False,
                                reverse_palette: bool = True):
    """Create a bar plot (horizontal by default) for feature importance.

    - feature_importance: pd.DataFrame or CSV path
    - feature_col, importance_col: column names containing labels and numeric importance
    - top_n: number of top features to keep (None for all)
    - sort_desc: sort by importance descending
    - normalize: scale importance to sum to 1 (optional)
    - cmap: str or list - color palette for bars. Can be a seaborn/matplotlib palette name (e.g., 'Blues', 'viridis') or a list of color hex codes. Default is 'Blues'.
    - save_path: optional file path where to save the figure
    - show: whether to call plt.show()
    - reverse_palette: if True, reverse the generated palette so higher importance bars are darker.
    - bar_color: hex color or Matplotlib color string to use for all bars (if set, overrides `cmap` and palette)
    Returns the matplotlib Axes object.
    """
    if isinstance(feature_importance, (str, Path)):
        fi = pd.read_csv(feature_importance)
    elif isinstance(feature_importance, pd.DataFrame):
        fi = feature_importance.copy()
    else:
        raise ValueError('feature_importance must be a DataFrame or path to CSV')

    if feature_col not in fi.columns:
        raise KeyError(f"feature column '{feature_col}' not found in DataFrame")
    if importance_col not in fi.columns:
        # attempt to find a numeric column automatically
        numeric_cols = fi.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise KeyError("No numeric column found for importance values")
        importance_col = numeric_cols[0]

    # Keep only relevant columns
    fi = fi[[hue_col, feature_col, importance_col]].dropna()
    #fi = fi.groupby(feature_col, as_index=False)[importance_col].mean()

    if normalize:
        total = fi[importance_col].sum()
        if total != 0:
            fi[importance_col] = fi[importance_col] / total

    if sort_desc:
        fi = fi.sort_values(importance_col, ascending=False)
    else:
        fi = fi.sort_values(importance_col, ascending=True)

    if top_n:
        fi = fi.head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Build palette dynamically so colors follow the number of bars and default to shades of blue
    n_colors = len(fi)
    # If a user passes a list/tuple, use it directly
    if isinstance(cmap, (list, tuple)):
        palette = cmap
    elif isinstance(cmap, str):
        try:
            palette = sns.color_palette(cmap, n_colors=n_colors)
        except Exception:
            # Fallback to a Blues palette if the provided cmap is invalid
            palette = sns.color_palette('Blues', n_colors=n_colors)
    else:
        palette = sns.color_palette('Blues', n_colors=n_colors)
    # Optionally reverse palette so bars reflect intensity in descending order
    if reverse_palette:
        palette = palette[::-1]

    # Configure axis appearance: white background, no grid
    ax.set_facecolor('white')
    ax.grid(False)
    sns.despine(ax=ax, offset=0)

    if horizontal:
        sns.barplot(x=importance_col, y=feature_col, data=fi, palette=palette, ax=ax,hue=hue_col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        sns.barplot(x=feature_col, y=importance_col, data=fi, palette=palette, ax=ax,hue=hue_col)
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        plt.xticks(rotation=45, ha='right')

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500)
    if show:
        plt.show()

    return ax


if __name__ == '__main__':
    # example usage: iterate config combinations and plot the top 10 features
    #sns.set_palette('Blues')

    for task, dimension, y_label in itertools.product(tasks, dimensions, y_labels):
        try:
            fi = load_feature_importance(results_dir, project_name, task, dimension, y_label, stat_folder, scoring, bootstrap_method,
                                         hyp_opt=hyp_opt, feature_selection=feature_selection,
                                         filename='feature_importance.csv')
        except FileNotFoundError as e:
            print(e)
            continue
        title = f"Feature importance {task} | {dimension} | {y_label}"
        # create figure and save to the results path for this combination
        folder_parts = [results_dir, project_name, 'plots', 'feature_importance', task, dimension, y_label]
        out_dir = _path_no_empty(*folder_parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'feature_importance.png'
        bar_plot_feature_importance(fi, top_n=5, normalize=False, title=title, save_path=out_file, show=False, cmap=["#ebb45c","#D973C7"],horizontal=True)
        print(f"Saved plot: {out_file}")
    
