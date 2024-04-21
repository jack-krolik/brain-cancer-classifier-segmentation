# Brain Cancer Classifier and Segmentation

## Datasets

The datasets used in this project are available on Kaggle:

- [Brain Tumor Image Dataset: Semantic Segmentation](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation)
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

## Setup Instructions

### Step 1: Install Dependencies

Install the necessary Python packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

Alternatively, you can create a conda environment using the following commands:

```bash
conda env create -f environment.yaml
conda activate ai_project_env
```

### Step 2: Setting Up Pre-commit Hooks

Our project uses pre-commit hooks to ensure the cleanliness and consistency of Jupyter notebooks by automatically stripping outputs before they are committed. This step helps maintain a clean git history and minimizes "diff noise."

After installing the project dependencies, activate the pre-commit hooks by running the following command:

```bash
pre-commit install
```

This command sets up the hooks based on our project's .pre-commit-config.yaml configuration and needs to be run only once.

This current hook cleans the Jupyter notebooks before they are committed.

### Step 3: Setup Environment Variables

To create a base configuration for the project, run the following command:

```bash
cp config/env_local.env .env
```

This will create a `.env` file in the root dir of the project. However, to actually run training and testing scripts, you will need to fill in the values in the `.env` file.

### Step 4: Kaggle API Authentication

Follow the instructions to set up your Kaggle API credentials. You can find the Kaggle API authentication instructions in the [Kaggle API Documentation](https://www.kaggle.com/docs/api).

### Step 5: Download Datasets

Refer to the `notebooks/downloading_datasets.ipynb` notebook for step-by-step instructions on using the Kaggle API to download the datasets required for this project. The datasets will be downloaded to the `./datasets` folder, which is configured to be ignored by git.

## Loading Classification Dataset

For an example of how to load the classification dataset, see the `notebooks/dataloader_example.ipynb` notebook. This notebook demonstrates how to use the `TumorClassificationDataset` class to load either the Training or Testing split from the [Tumor Classification Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Run Experiments:

### Classification

### Semantic Segmentation

See `src/scripts/train_segmentation.py` for logic related to running segmentation experiments. For more info run the following from the root directory to see available training configurations:
```bash
python -m src.scripts.train_segmentation --help
```

### Object Detection

Navigate to the `main` function in `src.scripts.train_object_detection.py` and edit the `training_configs`.
To run the specified experiment you can use...

```bash
python -m src.scripts.train_object_detection
```
