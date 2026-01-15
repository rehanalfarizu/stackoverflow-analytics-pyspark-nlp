"""
Chart Generator Module
======================
Generate berbagai jenis chart untuk analisis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from wordcloud import WordCloud
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ChartGenerator:
    """
    Generator untuk berbagai jenis chart.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize Chart Generator.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
    
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
        plt.Figure
            Matplotlib figure
        """
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
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if top_n:
            df = df.nlargest(top_n, y_column)
        
        if horizontal:
            ax.barh(df[x_column], df[y_column], color=self.colors[0])
            ax.set_xlabel(y_column)
            ax.set_ylabel(x_column)
        else:
            ax.bar(df[x_column], df[y_column], color=self.colors[0])
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
        plt.Figure
            Matplotlib figure
        """
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
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Pivot data
        pivot_df = df.pivot(index=y_column, columns=x_column, values=value_column)
        
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
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if columns:
            corr = df[columns].corr()
        else:
            corr = df.select_dtypes(include=[np.number]).corr()
        
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
        plt.Figure
            Matplotlib figure
        """
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
        plt.Figure
            Matplotlib figure
        """
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
