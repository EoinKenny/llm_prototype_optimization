import os
import zipfile
import shutil

from pathlib import Path


class DatasetUnzipper:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        # self.datasets = ["20newsgroups", "agnews", "amazon_reviews", "dbpedia", "imdb", 'trec']
        self.datasets = ['trec']
        
    def clean_directories(self):
        """Clean all dataset directories, keeping only zip files"""
        print("Cleaning directories (keeping zip files)...")
        for dataset in self.datasets:
            dataset_path = self.base_dir / dataset
            if dataset_path.exists():
                print(f"  Cleaning {dataset}...")
                
                # Get list of zip files to preserve
                zip_files = list(dataset_path.glob("*.zip"))
                zip_names = {zip_file.name for zip_file in zip_files}
                
                # Remove everything except zip files
                for item in dataset_path.iterdir():
                    if item.name not in zip_names:
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        except Exception as e:
                            print(f"    Warning: Could not remove {item.name}: {e}")
                
                print(f"    ✓ Cleaned {dataset}, preserved {len(zip_files)} zip files")

    def unzip_all(self):
        """Unzip all dataset folders with error handling"""
        for dataset in self.datasets:
            dataset_path = self.base_dir / dataset
            if dataset_path.exists():
                print(f"\nProcessing {dataset}...")
                for zip_file in dataset_path.glob("*.zip"):
                    print(f"  Extracting {zip_file.name}")
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            # Try to extract all at once first
                            zip_ref.extractall(dataset_path)
                            print(f"    ✓ Successfully extracted {zip_file.name}")
                    except (OSError, zipfile.BadZipFile) as e:
                        print(f"    ✗ Error extracting {zip_file.name}: {e}")
                        print(f"    Trying file-by-file extraction...")
                        
                        # Try extracting individual files
                        try:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                extracted_count = 0
                                total_files = len(zip_ref.namelist())
                                
                                for member in zip_ref.namelist():
                                    try:
                                        # Skip problematic files
                                        if len(member) > 200:  # Skip very long paths
                                            continue
                                        if any(char in member for char in ['<', '>', ':', '"', '|', '?', '*']):
                                            continue
                                            
                                        zip_ref.extract(member, dataset_path)
                                        extracted_count += 1
                                        
                                        if extracted_count % 1000 == 0:
                                            print(f"    Extracted {extracted_count}/{total_files} files...")
                                            
                                    except (OSError, zipfile.BadZipFile):
                                        # Skip individual problematic files
                                        continue
                                
                                print(f"    ✓ Extracted {extracted_count}/{total_files} files from {zip_file.name}")
                                
                        except Exception as final_error:
                            print(f"    ✗ Complete failure for {zip_file.name}: {final_error}")
                            print(f"    You may need to manually extract this file or re-download it")
                            continue
            else:
                print(f"  Warning: {dataset} folder not found")

    def run(self):
        """Clean and unzip all datasets"""
        print("="*50)
        print("DATASET UNZIPPER")
        print("="*50)
        
        self.clean_directories()
        self.unzip_all()
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print("Your datasets are ready for custom preprocessing!")

if __name__ == "__main__":
    unzipper = DatasetUnzipper()
    unzipper.run()