#!/bin/bash
# =============================================================================
# Stack Overflow Data Dump Downloader
# =============================================================================
# Script untuk mendownload dataset Stack Overflow dari Internet Archive
# Data Source: https://archive.org/details/stackexchange
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data/raw"
BASE_URL="https://archive.org/download/stackexchange"

# Files to download (Stack Overflow main site)
declare -a FILES=(
    "stackoverflow.com-Posts.7z"
    "stackoverflow.com-Users.7z"
    "stackoverflow.com-Comments.7z"
    "stackoverflow.com-Tags.7z"
    "stackoverflow.com-Votes.7z"
    "stackoverflow.com-Badges.7z"
    "stackoverflow.com-PostLinks.7z"
)

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v wget &> /dev/null; then
        print_error "wget is not installed. Please install it first."
        echo "  macOS: brew install wget"
        echo "  Ubuntu: sudo apt-get install wget"
        exit 1
    fi
    
    if ! command -v 7z &> /dev/null; then
        print_warning "7z is not installed. You'll need it to extract files."
        echo "  macOS: brew install p7zip"
        echo "  Ubuntu: sudo apt-get install p7zip-full"
    fi
    
    print_info "Dependencies check completed."
}

# Create data directory
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p "$DATA_DIR"
    mkdir -p "data/processed"
    mkdir -p "data/output"
    print_info "Directories created."
}

# Download a single file
download_file() {
    local filename=$1
    local url="${BASE_URL}/${filename}"
    local filepath="${DATA_DIR}/${filename}"
    
    if [ -f "$filepath" ]; then
        print_warning "$filename already exists. Skipping..."
        return 0
    fi
    
    print_info "Downloading $filename..."
    wget -c -P "$DATA_DIR" "$url" --progress=bar:force 2>&1
    
    if [ $? -eq 0 ]; then
        print_info "$filename downloaded successfully."
    else
        print_error "Failed to download $filename"
        return 1
    fi
}

# Extract 7z files
extract_files() {
    print_info "Extracting files..."
    
    for file in "$DATA_DIR"/*.7z; do
        if [ -f "$file" ]; then
            print_info "Extracting $(basename $file)..."
            7z x -o"$DATA_DIR" "$file" -y
        fi
    done
    
    print_info "Extraction completed."
}

# Download sample data (smaller dataset for testing)
download_sample() {
    print_info "Downloading sample dataset (smaller site)..."
    
    # Download a smaller Stack Exchange site for testing
    local sample_file="anime.stackexchange.com.7z"
    local url="${BASE_URL}/${sample_file}"
    
    wget -c -P "$DATA_DIR" "$url" --progress=bar:force 2>&1
    
    if [ -f "${DATA_DIR}/${sample_file}" ]; then
        7z x -o"${DATA_DIR}/sample" "${DATA_DIR}/${sample_file}" -y
        print_info "Sample data ready in ${DATA_DIR}/sample"
    fi
}

# Main execution
main() {
    echo "=============================================="
    echo "  Stack Overflow Data Dump Downloader"
    echo "=============================================="
    echo ""
    
    check_dependencies
    setup_directories
    
    echo ""
    echo "Select download option:"
    echo "  1) Download full Stack Overflow dataset (~100GB+)"
    echo "  2) Download sample dataset (~500MB)"
    echo "  3) Download specific file"
    echo "  4) Extract existing files"
    echo ""
    
    read -p "Enter your choice [1-4]: " choice
    
    case $choice in
        1)
            print_info "Starting full dataset download..."
            print_warning "This will download approximately 100GB+ of data."
            read -p "Are you sure you want to continue? [y/N]: " confirm
            
            if [[ $confirm =~ ^[Yy]$ ]]; then
                for file in "${FILES[@]}"; do
                    download_file "$file"
                done
                
                read -p "Extract files now? [y/N]: " extract_confirm
                if [[ $extract_confirm =~ ^[Yy]$ ]]; then
                    extract_files
                fi
            fi
            ;;
        2)
            download_sample
            ;;
        3)
            echo "Available files:"
            for i in "${!FILES[@]}"; do
                echo "  $((i+1))) ${FILES[$i]}"
            done
            read -p "Enter file number: " file_num
            
            if [[ $file_num -ge 1 && $file_num -le ${#FILES[@]} ]]; then
                download_file "${FILES[$((file_num-1))]}"
            else
                print_error "Invalid selection."
            fi
            ;;
        4)
            extract_files
            ;;
        *)
            print_error "Invalid option."
            exit 1
            ;;
    esac
    
    echo ""
    print_info "Done!"
}

# Run main function
main "$@"
