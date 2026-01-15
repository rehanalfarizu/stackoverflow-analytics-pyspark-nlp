# Makefile for Stack Overflow Analytics
# ======================================

.PHONY: help install clean test run etl nlp ml dashboard lint format

# Default target
help:
	@echo "Stack Overflow Analytics - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install all dependencies"
	@echo "  make setup       Complete setup (install + download sample)"
	@echo ""
	@echo "Run Pipeline:"
	@echo "  make run         Run full pipeline"
	@echo "  make etl         Run ETL only"
	@echo "  make nlp         Run NLP analysis only"
	@echo "  make ml          Run ML training only"
	@echo "  make dashboard   Launch Streamlit dashboard"
	@echo ""
	@echo "Data:"
	@echo "  make download    Download sample dataset"
	@echo "  make download-full  Download full dataset (100GB+)"
	@echo ""
	@echo "Development:"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linter"
	@echo "  make format      Format code"
	@echo "  make clean       Clean generated files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	python -m nltk.downloader punkt stopwords wordnet vader_lexicon
	python -m spacy download en_core_web_sm || true

# Setup everything
setup: install download
	@echo "Setup complete!"

# Download sample data
download:
	@chmod +x scripts/download_data.sh
	@./scripts/download_data.sh

# Download full dataset
download-full:
	@echo "WARNING: This will download 100GB+ of data"
	@chmod +x scripts/download_data.sh
	@./scripts/download_data.sh

# Run full pipeline
run:
	python main.py --mode full

# Run ETL only
etl:
	python main.py --mode etl

# Run NLP only
nlp:
	python main.py --mode nlp

# Run ML only
ml:
	python main.py --mode ml

# Run dashboard
dashboard:
	streamlit run src/visualization/dashboard.py

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Run linter
lint:
	flake8 src/ tests/ --max-line-length=100
	isort --check-only src/ tests/

# Format code
format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf data/processed/* data/output/*
	rm -rf logs/*.log
	rm -rf models/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Create sample notebook
notebook:
	jupyter notebook notebooks/

# Docker build
docker-build:
	docker build -t stackoverflow-analytics .

# Docker run
docker-run:
	docker run -p 8501:8501 stackoverflow-analytics
