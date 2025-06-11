# PowerShell script to create the project directory structure.

Write-Host "Starting project setup..."

# --- Create Directories ---
# An array holding all the directory paths we need.
$directories = @(
    "data/raw",
    "data/processed",
    "figures",
    "models",
    "notebooks",
    "src",
    "streamlit_app/sample_images"
)

# Loop through the array and create each directory if it doesn't already exist.
$directories | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Force -Path $_
        Write-Host "Created Directory: $_"
    } else {
        Write-Host "Directory already exists: $_"
    }
}

# --- Create Placeholder Files ---
# An array holding all the placeholder files to create.
$files = @(
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "figures/.gitkeep",
    "models/.gitkeep",
    "src/.gitkeep",
    "streamlit_app/sample_images/.gitkeep",
    "notebooks/01_data_understanding_and_eda.ipynb",
    "notebooks/02_preprocessing_and_augmentation.ipynb",
    "notebooks/03_model_training_and_benchmarking.ipynb",
    "notebooks/04_inference_and_app_preparation.ipynb",
    "src/data_utils.py",
    "src/models.py",
    "src/metrics.py",
    "streamlit_app/app.py"
)

# Loop through the array and create each empty file.
$files | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType File -Path $_
        Write-Host "Created File: $_"
    } else {
        Write-Host "File already exists: $_"
    }
}

Write-Host "`nProject structure setup complete."