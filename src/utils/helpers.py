"""
Helper Functions
================
Berbagai fungsi helper untuk proyek.
"""

import os
import yaml
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib


def load_config(config_path: str = "config/settings.yaml") -> Dict:
    """
    Load konfigurasi dari file YAML.
    
    Parameters
    ----------
    config_path : str
        Path ke file konfigurasi
        
    Returns
    -------
    Dict
        Konfigurasi
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Menyimpan data ke file JSON.
    
    Parameters
    ----------
    data : Any
        Data untuk disimpan
    filepath : str
        Path output
    indent : int
        Indentasi
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """
    Load data dari file JSON.
    
    Parameters
    ----------
    filepath : str
        Path file
        
    Returns
    -------
    Any
        Data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(path: str) -> str:
    """
    Memastikan direktori ada, buat jika tidak ada.
    
    Parameters
    ----------
    path : str
        Path direktori
        
    Returns
    -------
    str
        Path yang sama
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """
    Mendapatkan hash dari file.
    
    Parameters
    ----------
    filepath : str
        Path file
    algorithm : str
        Algoritma hash (md5, sha256)
        
    Returns
    -------
    str
        Hash string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def format_number(num: float) -> str:
    """
    Format angka untuk display (1000 -> 1K, 1000000 -> 1M).
    
    Parameters
    ----------
    num : float
        Angka
        
    Returns
    -------
    str
        Formatted string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(int(num))


def format_duration(seconds: float) -> str:
    """
    Format durasi dalam seconds ke string yang readable.
    
    Parameters
    ----------
    seconds : float
        Durasi dalam detik
        
    Returns
    -------
    str
        Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def get_timestamp() -> str:
    """
    Mendapatkan timestamp saat ini.
    
    Returns
    -------
    str
        Timestamp string (YYYY-MM-DD_HH-MM-SS)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_date_string() -> str:
    """
    Mendapatkan tanggal saat ini.
    
    Returns
    -------
    str
        Date string (YYYY-MM-DD)
    """
    return datetime.now().strftime("%Y-%m-%d")


def parse_tags(tags_string: str) -> List[str]:
    """
    Parse tags dari format Stack Overflow (<tag1><tag2>).
    
    Parameters
    ----------
    tags_string : str
        String tags
        
    Returns
    -------
    List[str]
        List of tags
    """
    import re
    if not tags_string:
        return []
    return re.findall(r'<([^>]+)>', tags_string)


def clean_html(text: str) -> str:
    """
    Menghapus HTML tags dari teks.
    
    Parameters
    ----------
    text : str
        Teks dengan HTML
        
    Returns
    -------
    str
        Clean text
    """
    import re
    if not text:
        return ""
    
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean)
    
    return clean.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text ke panjang maksimum.
    
    Parameters
    ----------
    text : str
        Teks
    max_length : int
        Panjang maksimum
    suffix : str
        Suffix untuk text yang dipotong
        
    Returns
    -------
    str
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def calculate_percentage(part: float, whole: float, decimals: int = 2) -> float:
    """
    Menghitung persentase.
    
    Parameters
    ----------
    part : float
        Bagian
    whole : float
        Total
    decimals : int
        Jumlah desimal
        
    Returns
    -------
    float
        Persentase
    """
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, decimals)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Membagi list menjadi chunks.
    
    Parameters
    ----------
    lst : List
        Input list
    chunk_size : int
        Ukuran chunk
        
    Returns
    -------
    List[List]
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten nested list.
    
    Parameters
    ----------
    nested_list : List[List]
        Nested list
        
    Returns
    -------
    List
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division yang mengembalikan default jika denominator 0.
    
    Parameters
    ----------
    numerator : float
        Pembilang
    denominator : float
        Penyebut
    default : float
        Nilai default jika divide by zero
        
    Returns
    -------
    float
        Hasil pembagian
    """
    if denominator == 0:
        return default
    return numerator / denominator


class Timer:
    """Context manager untuk timing."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} completed in {format_duration(duration)}")
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class ProgressTracker:
    """Tracker untuk progress operasi."""
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        self.current += n
        self._print_progress()
    
    def _print_progress(self) -> None:
        percentage = calculate_percentage(self.current, self.total)
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0
        
        print(f"\r{self.name}: {self.current}/{self.total} "
              f"({percentage}%) - ETA: {format_duration(eta)}", end="")
        
        if self.current >= self.total:
            print()  # New line when complete
