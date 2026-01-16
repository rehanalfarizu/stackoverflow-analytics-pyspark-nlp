"""
Chart Generator Module
======================
Generate berbagai jenis chart untuk analisis.
Kompatibel dengan Google Colab dan environment lokal.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from wordcloud import WordCloud
import logging
import warnings

logger = logging.getLogger(__name__)


def is_running_in_colab() -> bool:
    """
    Cek apakah sedang berjalan di Google Colab.
    
    Returns
    -------
    bool
        True jika di Google Colab
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_display():
    """
    Setup display untuk Google Colab.
    Mengaktifkan inline plotting dan high DPI.
    """
    if is_running_in_colab():
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        
        # Set retina display untuk kualitas lebih baik
        from IPython.display import set_matplotlib_formats
        try:
            set_matplotlib_formats('retina')
        except:
            pass
        
        # Increase figure DPI
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        
        logger.info("Google Colab display mode activated")


def install_colab_dependencies():
    """
    Install dependencies yang diperlukan di Google Colab.
    Jalankan ini di awal notebook Colab.
    """
    if is_running_in_colab():
        import subprocess
        import sys
        
        packages = ['wordcloud', 'plotly']
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
                print(f"Installed {package}")


# Check for Plotly availability (better for Colab interactivity)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive charts will not be available. "
                  "Install with: pip install plotly")

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

# Auto-setup for Colab
if is_running_in_colab():
    setup_colab_display()


class ChartGenerator:
    """
    Generator untuk berbagai jenis chart.
    Kompatibel dengan Google Colab dan environment lokal.
    Mendukung chart statis (Matplotlib) dan interaktif (Plotly).
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), interactive: bool = None):
        """
        Initialize Chart Generator.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Default figure size
        interactive : bool, optional
            Gunakan Plotly untuk chart interaktif.
            Default: True jika di Colab dan Plotly tersedia
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        self.hex_colors = [self._rgb_to_hex(c) for c in self.colors]
        
        # Auto-detect: use interactive in Colab if available
        if interactive is None:
            self.interactive = is_running_in_colab() and PLOTLY_AVAILABLE
        else:
            self.interactive = interactive and PLOTLY_AVAILABLE
        
        if is_running_in_colab():
            logger.info("ChartGenerator initialized for Google Colab")
    
    def _rgb_to_hex(self, rgb: Tuple) -> str:
        """Convert RGB tuple to hex color."""
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    
    def show(self, fig) -> None:
        """
        Display figure - kompatibel dengan Colab dan lokal.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Figure to display
        """
        if is_running_in_colab():
            if hasattr(fig, 'show'):  # Plotly figure
                fig.show()
            else:  # Matplotlib figure
                plt.show()
        else:
            if hasattr(fig, 'show'):
                fig.show()
            else:
                plt.show()
    
    def plot_trend(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        group_column: str = None,
        title: str = "Trend Analysis",
        xlabel: str = None,
        ylabel: str = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot trend line chart.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data untuk plotting
        x_column : str
            Kolom untuk x-axis
        y_column : str
            Kolom untuk y-axis
        group_column : str, optional
            Kolom untuk grouping (multiple lines)
        title : str
            Judul chart
        xlabel : str, optional
            Label x-axis
        ylabel : str, optional
            Label y-axis
        save_path : str, optional
            Path untuk menyimpan chart
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            if group_column:
                fig = px.line(
                    df, x=x_column, y=y_column, color=group_column,
                    title=title, labels={x_column: xlabel or x_column, y_column: ylabel or y_column}
                )
            else:
                fig = px.line(
                    df, x=x_column, y=y_column,
                    title=title, labels={x_column: xlabel or x_column, y_column: ylabel or y_column}
                )
            
            fig.update_layout(
                width=self.figsize[0] * 80,
                height=self.figsize[1] * 80,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
                logger.info(f"Chart saved to {save_path}")
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if group_column:
            for idx, group in enumerate(df[group_column].unique()):
                group_data = df[df[group_column] == group]
                ax.plot(
                    group_data[x_column],
                    group_data[y_column],
                    label=group,
                    color=self.colors[idx % len(self.colors)],
                    linewidth=2
                )
            ax.legend(loc='best')
        else:
            ax.plot(df[x_column], df[y_column], linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or y_column)
        
        # Format x-axis if datetime
        if pd.api.types.is_datetime64_any_dtype(df[x_column]):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def plot_bar(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Bar Chart",
        horizontal: bool = False,
        top_n: int = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot bar chart.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data untuk plotting
        x_column : str
            Kolom untuk kategori
        y_column : str
            Kolom untuk nilai
        title : str
            Judul chart
        horizontal : bool
            Jika True, bar horizontal
        top_n : int, optional
            Hanya tampilkan top N
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        plot_df = df.copy()
        if top_n:
            plot_df = plot_df.nlargest(top_n, y_column)
        
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            if horizontal:
                fig = px.bar(
                    plot_df, x=y_column, y=x_column, orientation='h',
                    title=title, color_discrete_sequence=self.hex_colors
                )
            else:
                fig = px.bar(
                    plot_df, x=x_column, y=y_column,
                    title=title, color_discrete_sequence=self.hex_colors
                )
            
            fig.update_layout(
                width=self.figsize[0] * 80,
                height=self.figsize[1] * 80,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if horizontal:
            ax.barh(plot_df[x_column], plot_df[y_column], color=self.colors[0])
            ax.set_xlabel(y_column)
            ax.set_ylabel(x_column)
        else:
            ax.bar(plot_df[x_column], plot_df[y_column], color=self.colors[0])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_pie(
        self,
        df: pd.DataFrame,
        values_column: str,
        labels_column: str,
        title: str = "Distribution",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot pie chart.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data untuk plotting
        values_column : str
            Kolom nilai
        labels_column : str
            Kolom label
        title : str
            Judul
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            fig = px.pie(
                df, values=values_column, names=labels_column,
                title=title, color_discrete_sequence=self.hex_colors
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                width=600,
                height=600,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.pie(
            df[values_column],
            labels=df[labels_column],
            autopct='%1.1f%%',
            colors=self.colors[:len(df)],
            startangle=90
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        value_column: str,
        title: str = "Heatmap",
        cmap: str = "YlOrRd",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot heatmap.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data dengan kolom x, y, dan value
        x_column : str
            Kolom untuk x-axis
        y_column : str
            Kolom untuk y-axis
        value_column : str
            Kolom untuk nilai warna
        title : str
            Judul
        cmap : str
            Colormap
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        # Pivot data
        pivot_df = df.pivot(index=y_column, columns=x_column, values=value_column)
        
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            fig = px.imshow(
                pivot_df,
                title=title,
                color_continuous_scale=cmap.lower(),
                text_auto=True
            )
            fig.update_layout(
                width=self.figsize[0] * 80,
                height=self.figsize[1] * 80,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            pivot_df,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt='.0f',
            linewidths=0.5
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_wordcloud(
        self,
        text: str = None,
        word_freq: Dict[str, int] = None,
        title: str = "Word Cloud",
        max_words: int = 100,
        save_path: str = None
    ) -> plt.Figure:
        """
        Generate word cloud.
        
        Parameters
        ----------
        text : str, optional
            Teks untuk generate wordcloud
        word_freq : Dict[str, int], optional
            Dictionary word frequency
        title : str
            Judul
        max_words : int
            Maksimum kata
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if word_freq:
            wc = WordCloud(
                width=800,
                height=400,
                max_words=max_words,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
        elif text:
            wc = WordCloud(
                width=800,
                height=400,
                max_words=max_words,
                background_color='white',
                colormap='viridis'
            ).generate(text)
        else:
            raise ValueError("Either text or word_freq must be provided")
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        title: str = "Correlation Matrix",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot correlation matrix.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data
        columns : List[str], optional
            Kolom untuk korelasi
        title : str
            Judul
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        if columns:
            corr = df[columns].corr()
        else:
            corr = df.select_dtypes(include=[np.number]).corr()
        
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            corr_masked = corr.copy()
            corr_masked.values[mask] = np.nan
            
            fig = px.imshow(
                corr_masked,
                title=title,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                text_auto='.2f'
            )
            fig.update_layout(
                width=700,
                height=600,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr,
            mask=mask,
            ax=ax,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            linewidths=0.5
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_distribution(
        self,
        df: pd.DataFrame,
        column: str,
        title: str = "Distribution",
        bins: int = 30,
        kde: bool = True,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot distribution histogram.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data
        column : str
            Kolom untuk plot
        title : str
            Judul
        bins : int
            Jumlah bins
        kde : bool
            Tambahkan KDE curve
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            fig = px.histogram(
                df, x=column, nbins=bins,
                title=title, marginal='box' if kde else None,
                color_discrete_sequence=self.hex_colors
            )
            fig.update_layout(
                width=self.figsize[0] * 80,
                height=self.figsize[1] * 80,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.histplot(
            df[column],
            ax=ax,
            bins=bins,
            kde=kde,
            color=self.colors[0]
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_boxplot(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Box Plot",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot boxplot.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data
        x_column : str
            Kolom kategori
        y_column : str
            Kolom nilai
        title : str
            Judul
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            fig = px.box(
                df, x=x_column, y=y_column,
                title=title, color=x_column,
                color_discrete_sequence=self.hex_colors
            )
            fig.update_layout(
                width=self.figsize[0] * 80,
                height=self.figsize[1] * 80,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.boxplot(
            data=df,
            x=x_column,
            y=y_column,
            ax=ax,
            palette=self.colors
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(
        self,
        charts: List[Tuple[plt.Figure, str]],
        suptitle: str = "Analytics Dashboard",
        save_path: str = None
    ) -> plt.Figure:
        """
        Membuat dashboard dengan multiple charts.
        
        Parameters
        ----------
        charts : List[Tuple[plt.Figure, str]]
            List of (figure, title) tuples
        suptitle : str
            Main title
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure
            Combined figure
        """
        n_charts = len(charts)
        cols = min(2, n_charts)
        rows = (n_charts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
        
        if n_charts == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Note: This is a simplified version
        # In practice, you would recreate each chart in the subplot
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_scatter(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: str = None,
        size_column: str = None,
        title: str = "Scatter Plot",
        xlabel: str = None,
        ylabel: str = None,
        save_path: str = None
    ):
        """
        Plot scatter chart - ideal untuk Colab dengan interaktivitas.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data untuk plotting
        x_column : str
            Kolom untuk x-axis
        y_column : str
            Kolom untuk y-axis
        color_column : str, optional
            Kolom untuk warna points
        size_column : str, optional
            Kolom untuk ukuran points
        title : str
            Judul chart
        xlabel : str, optional
            Label x-axis
        ylabel : str, optional
            Label y-axis
        save_path : str, optional
            Path untuk menyimpan
            
        Returns
        -------
        plt.Figure or plotly.graph_objects.Figure
            Figure object
        """
        # Interactive mode dengan Plotly
        if self.interactive and PLOTLY_AVAILABLE:
            fig = px.scatter(
                df, x=x_column, y=y_column,
                color=color_column, size=size_column,
                title=title,
                labels={x_column: xlabel or x_column, y_column: ylabel or y_column},
                color_discrete_sequence=self.hex_colors
            )
            fig.update_layout(
                width=self.figsize[0] * 80,
                height=self.figsize[1] * 80,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path)
            
            return fig
        
        # Static mode dengan Matplotlib
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if color_column:
            scatter = ax.scatter(
                df[x_column], df[y_column],
                c=pd.Categorical(df[color_column]).codes,
                s=df[size_column] if size_column else 50,
                alpha=0.6, cmap='husl'
            )
            plt.colorbar(scatter, label=color_column)
        else:
            ax.scatter(
                df[x_column], df[y_column],
                s=df[size_column] if size_column else 50,
                alpha=0.6, color=self.colors[0]
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or y_column)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# ============================================================================
# GOOGLE COLAB HELPER FUNCTIONS
# ============================================================================

def colab_quick_setup():
    """
    Quick setup untuk Google Colab.
    Jalankan di awal notebook Colab.
    
    Example
    -------
    >>> from src.visualization.charts import colab_quick_setup
    >>> colab_quick_setup()
    """
    if not is_running_in_colab():
        print("Not running in Google Colab")
        return
    
    install_colab_dependencies()
    setup_colab_display()
    print("✅ Google Colab visualization setup complete!")
    print("   - Matplotlib inline mode activated")
    print("   - High DPI display enabled")
    print("   - Dependencies installed")


def create_colab_charts(interactive: bool = True) -> ChartGenerator:
    """
    Factory function untuk membuat ChartGenerator yang optimal untuk Colab.
    
    Parameters
    ----------
    interactive : bool
        Gunakan Plotly untuk chart interaktif
    
    Returns
    -------
    ChartGenerator
        Instance yang sudah dikonfigurasi untuk Colab
    
    Example
    -------
    >>> from src.visualization.charts import create_colab_charts
    >>> charts = create_colab_charts()
    >>> fig = charts.plot_trend(df, 'date', 'value')
    >>> charts.show(fig)
    """
    return ChartGenerator(figsize=(12, 6), interactive=interactive)


def save_chart_to_drive(fig, filename: str, folder: str = "/content/drive/MyDrive/charts"):
    """
    Simpan chart ke Google Drive (khusus Colab).
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure untuk disimpan
    filename : str
        Nama file (dengan ekstensi, misal 'chart.png')
    folder : str
        Folder tujuan di Google Drive
    
    Example
    -------
    >>> from src.visualization.charts import save_chart_to_drive
    >>> save_chart_to_drive(fig, 'trend_analysis.png')
    """
    if not is_running_in_colab():
        print("This function is only available in Google Colab")
        return
    
    import os
    
    # Mount Google Drive jika belum
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return
    
    # Buat folder jika tidak ada
    os.makedirs(folder, exist_ok=True)
    
    filepath = os.path.join(folder, filename)
    
    if hasattr(fig, 'write_image'):  # Plotly
        fig.write_image(filepath)
    else:  # Matplotlib
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    print(f"✅ Chart saved to: {filepath}")


def display_side_by_side(*figs, titles: List[str] = None):
    """
    Display multiple figures side by side di Colab.
    
    Parameters
    ----------
    *figs : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figures untuk ditampilkan
    titles : List[str], optional
        Judul untuk setiap figure
    
    Example
    -------
    >>> from src.visualization.charts import display_side_by_side
    >>> display_side_by_side(fig1, fig2, titles=['Chart 1', 'Chart 2'])
    """
    if is_running_in_colab():
        from IPython.display import display, HTML
        
        if titles is None:
            titles = [f"Chart {i+1}" for i in range(len(figs))]
        
        for title, fig in zip(titles, figs):
            print(f"\n{'='*50}")
            print(f"📊 {title}")
            print('='*50)
            
            if hasattr(fig, 'show'):  # Plotly
                fig.show()
            else:  # Matplotlib
                plt.figure(fig.number)
                plt.show()
    else:
        for fig in figs:
            if hasattr(fig, 'show'):
                fig.show()
            else:
                plt.show()
