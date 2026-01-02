import kagglehub
import pandas as pd
import os


def log_to_file(content, mode='a'):
    with open("analysis_report.md", mode, encoding='utf-8') as f:
        f.write(content + "\n")

def explore_dataset():
    log_to_file("# Dataset Exploration Report", 'w')
    print("Downloading dataset...")
    try:
        path = kagglehub.dataset_download("haseebindata/student-performance-predictions")
        log_to_file(f"Dataset downloaded to: {path}")
    except Exception as e:
        log_to_file(f"Error downloading dataset: {e}")
        return

    log_to_file("\n## CSV Files")
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
                log_to_file(f"- {full_path}")

    if not csv_files:
        log_to_file("No CSV files found.")
        return

    for csv_file in csv_files:
        log_to_file(f"\n## Analysis of {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            log_to_file("\n### Columns")
            log_to_file(", ".join(list(df.columns)))
            
            log_to_file("\n### First 5 Rows")
            log_to_file(df.head().to_markdown())
            
            log_to_file("\n### Basic Statistics")
            log_to_file(df.describe(include='all').to_markdown())
            
            # Simple inference of column types
            log_to_file("\n### Column Types")
            log_to_file(str(df.dtypes))

        except Exception as e:
            log_to_file(f"Error reading {csv_file}: {e}")

if __name__ == "__main__":
    explore_dataset()

