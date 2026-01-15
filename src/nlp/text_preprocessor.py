"""
Text Preprocessor Module
========================
Preprocessing teks untuk analisis NLP.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram
from typing import List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Preprocessor untuk teks Stack Overflow.
    
    Melakukan:
    - Pembersihan HTML
    - Penghapusan code blocks
    - Tokenisasi
    - Penghapusan stopwords
    - Lemmatization (opsional)
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Text Preprocessor.
        
        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        """
        self.spark = spark
        self._register_udfs()
        
        # Technical stopwords untuk Stack Overflow
        self.tech_stopwords = [
            'use', 'using', 'used', 'want', 'need', 'try', 'trying',
            'get', 'getting', 'make', 'making', 'work', 'working',
            'code', 'error', 'problem', 'issue', 'question', 'answer',
            'help', 'please', 'thank', 'thanks', 'anyone', 'someone',
            'way', 'thing', 'something', 'anything', 'nothing',
            'would', 'could', 'should', 'might', 'must', 'like',
            'know', 'think', 'look', 'see', 'seem', 'find', 'give'
        ]
    
    def _register_udfs(self):
        """Register User Defined Functions."""
        
        @F.udf(returnType=StringType())
        def clean_text(text):
            if text is None:
                return ""
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Remove code blocks
            text = re.sub(r'```[\s\S]*?```', ' ', text)
            text = re.sub(r'`[^`]+`', ' ', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', ' ', text)
            
            # Remove special characters but keep alphanumeric
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Remove numbers
            text = re.sub(r'\d+', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip().lower()
        
        self._clean_text_udf = clean_text
        
        @F.udf(returnType=ArrayType(StringType()))
        def simple_tokenize(text):
            if text is None:
                return []
            # Simple whitespace tokenization
            tokens = text.lower().split()
            # Filter tokens: minimal 2 karakter, maksimal 50
            return [t for t in tokens if 2 <= len(t) <= 50]
        
        self._tokenize_udf = simple_tokenize
        
        @F.udf(returnType=ArrayType(StringType()))
        def filter_tokens(tokens, min_len=2, max_len=30):
            if tokens is None:
                return []
            return [t for t in tokens if min_len <= len(t) <= max_len]
        
        self._filter_tokens_udf = filter_tokens
    
    def preprocess(
        self,
        df: DataFrame,
        text_column: str = "Body",
        output_column: str = "ProcessedText",
        remove_stopwords: bool = True
    ) -> DataFrame:
        """
        Preprocess teks lengkap.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        text_column : str
            Nama kolom teks
        output_column : str
            Nama kolom output
        remove_stopwords : bool
            Hapus stopwords
            
        Returns
        -------
        DataFrame
            DataFrame dengan teks terproses
        """
        logger.info(f"Preprocessing text column: {text_column}")
        
        # Clean text
        cleaned_col = f"{output_column}_cleaned"
        df = df.withColumn(cleaned_col, self._clean_text_udf(F.col(text_column)))
        
        # Tokenize
        tokenized_col = f"{output_column}_tokens"
        tokenizer = Tokenizer(inputCol=cleaned_col, outputCol=tokenized_col)
        df = tokenizer.transform(df)
        
        # Remove stopwords
        if remove_stopwords:
            # Get default English stopwords
            remover = StopWordsRemover(
                inputCol=tokenized_col, 
                outputCol=f"{output_column}_filtered"
            )
            # Add custom technical stopwords
            all_stopwords = remover.getStopWords() + self.tech_stopwords
            remover.setStopWords(all_stopwords)
            df = remover.transform(df)
            
            # Rename to final column
            df = df.withColumn(output_column, F.col(f"{output_column}_filtered"))
        else:
            df = df.withColumn(output_column, F.col(tokenized_col))
        
        # Clean up intermediate columns
        df = df.drop(cleaned_col, tokenized_col, f"{output_column}_filtered")
        
        logger.info("Text preprocessing completed")
        return df
    
    def preprocess_title(
        self,
        df: DataFrame,
        title_column: str = "Title",
        output_column: str = "ProcessedTitle"
    ) -> DataFrame:
        """
        Preprocess kolom title.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        title_column : str
            Nama kolom title
        output_column : str
            Nama kolom output
            
        Returns
        -------
        DataFrame
            DataFrame dengan title terproses
        """
        return self.preprocess(
            df, 
            text_column=title_column, 
            output_column=output_column
        )
    
    def create_ngrams(
        self,
        df: DataFrame,
        input_column: str = "ProcessedText",
        n: int = 2
    ) -> DataFrame:
        """
        Membuat n-grams dari tokens.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame dengan tokens
        input_column : str
            Kolom tokens
        n : int
            Ukuran n-gram (2 = bigram, 3 = trigram)
            
        Returns
        -------
        DataFrame
            DataFrame dengan kolom n-grams
        """
        ngram = NGram(n=n, inputCol=input_column, outputCol=f"{input_column}_{n}grams")
        return ngram.transform(df)
    
    def extract_code_snippets(
        self,
        df: DataFrame,
        body_column: str = "Body"
    ) -> DataFrame:
        """
        Ekstrak code snippets dari body.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        body_column : str
            Kolom body HTML
            
        Returns
        -------
        DataFrame
            DataFrame dengan kolom CodeSnippets
        """
        @F.udf(returnType=ArrayType(StringType()))
        def extract_code(text):
            if text is None:
                return []
            
            # Extract <code> blocks
            code_blocks = re.findall(r'<code>(.*?)</code>', text, re.DOTALL)
            
            # Extract ``` blocks  
            md_blocks = re.findall(r'```[\w]*\n?([\s\S]*?)```', text)
            
            return code_blocks + md_blocks
        
        return df.withColumn("CodeSnippets", extract_code(F.col(body_column)))
    
    def calculate_text_features(self, df: DataFrame) -> DataFrame:
        """
        Menghitung fitur-fitur teks.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame dengan ProcessedText
            
        Returns
        -------
        DataFrame
            DataFrame dengan fitur tambahan
        """
        return df \
            .withColumn("NumTokens", F.size("ProcessedText")) \
            .withColumn("AvgTokenLength", 
                F.aggregate(
                    "ProcessedText",
                    F.lit(0.0),
                    lambda acc, x: acc + F.length(x),
                    lambda acc: acc / F.greatest(F.size("ProcessedText"), F.lit(1))
                )
            ) \
            .withColumn("UniqueTokenRatio",
                F.size(F.array_distinct("ProcessedText")) / 
                F.greatest(F.size("ProcessedText"), F.lit(1))
            )
    
    def combine_text_columns(
        self,
        df: DataFrame,
        columns: List[str],
        output_column: str = "CombinedText",
        separator: str = " "
    ) -> DataFrame:
        """
        Menggabungkan beberapa kolom teks.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        columns : List[str]
            Kolom untuk digabung
        output_column : str
            Nama kolom output
        separator : str
            Separator antar kolom
            
        Returns
        -------
        DataFrame
            DataFrame dengan kolom gabungan
        """
        return df.withColumn(
            output_column,
            F.concat_ws(separator, *[F.col(c) for c in columns])
        )
