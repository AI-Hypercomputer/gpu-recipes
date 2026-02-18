# Data preprocessing scripts for Waymo datasets
This page introduces how to use the provided example scripts to process the Waymo datasets for recipes in this repository.

## Data Preprocessing for Waymo Perception Dataset

This document provides instructions for using the `waymo_perception_data_processor.py` script to download, process, and format the Waymo Open Perception Dataset into a Hugging Face Dataset.

### 1. Introduction to the Waymo Perception Dataset

The Waymo Open Dataset offers large-scale, high-quality data for autonomous driving research. This script is designed to work with the Perception dataset (v2.0.1), which includes high-resolution sensor data and corresponding labels.

This processor specifically handles:
- **Camera Images**: Extracts images from the five cameras (front, front-left, front-right, side-left, side-right).
- **2D Bounding Boxes**: Uses the 2D bounding box labels to generate question-answer pairs about the objects present in each image.

### Accessing the Dataset

To use this script, you must first get access to the Waymo Open Dataset.

1.  Visit the Waymo Open Dataset website and register to get access.
2.  Ensure you have a Google Cloud account and are authenticated. The script requires access to the dataset's GCS bucket: `gs://waymo_open_dataset_v_2_0_1/`.

### 2. Prerequisites

Before running the script, ensure you have the following prerequisites installed and configured.

#### Google Cloud SDK

The `gsutil` command-line tool is required to download the dataset from Google Cloud Storage.

1.  Install the Google Cloud SDK.
2.  Authenticate with Google Cloud:
    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

#### Python Packages

The script relies on several Python libraries. You can install them using pip:

```bash
pip install pandas datasets pyarrow click Pillow
```

### 3. How to Run the Script

The script is a command-line tool that processes the dataset in parallel.

#### Command-Line Arguments

The script accepts the following arguments:

| Argument                 | Alias | Default Value                 | Description                                                              |
| :----------------------- | :---- | :---------------------------- | :----------------------------------------------------------------------- |
| `--output_dir`           | `-o`  | `./waymo_processed_dataset`   | The directory where the final processed Parquet files will be saved.     |
| `--input_dir`            | `-i`  | `./waymo_data`                | The local directory to download the raw Waymo dataset files into.        |
| `--num_threads`          | `-n`  | 8                             | The number of threads to use for parallel processing of data segments.   |
| `--output_filename_base` |       | `waymo_processed_dataset`     | The base name for the output Parquet files.                              |
| `--download`             | `-d`  | (flag)                        | If specified, the script will download the dataset from GCS.             |

#### Example Usage

1.  **Download and Process the Dataset**

    This command will first download the Waymo camera image and box data into the `./waymo_data` directory and then process it, saving the result in `./waymo_processed_dataset`.

    ```bash
    python waymo_perception_data_processor.py --download --num_threads 16
    ```

2.  **Process Locally Stored Data**

    If you have already downloaded the data, you can run the processing step directly by omitting the `--download` flag.

    ```bash
    python waymo_perception_data_processor.py --input_dir ./path/to/waymo_data --output_dir ./my_output --num_threads 16
    ```

### 4. Processing Result

The script generates multiple Parquet files in the specified output directory. Each file corresponds to a processed segment from the original dataset. These Parquet files can be easily loaded as a Hugging Face Dataset.

Each row in the resulting dataset contains the following columns:

-   **`image`**: A `PIL.Image` object (resized to 250x200) for the camera frame.
-   **`question`**: A `string` asking what objects are visible in the image.
-   **`multiple_choice_answer`**: A `string` listing the detected objects (e.g., `Objects: TYPE_CAR, TYPE_PEDESTRIAN.`) or stating that no common objects were detected.

#### Loading the Processed Data

You can load the processed data using the `datasets` library:

```python
from datasets import load_dataset

# Load all Parquet files from the output directory
processed_dataset = load_dataset("parquet", data_files="waymo_processed_dataset/*.parquet")

print(processed_dataset)
print(processed_dataset[0])
```

### 5. Common Issues

1.  **`gsutil` Command Not Found**: This error occurs if the Google Cloud SDK is not installed or not in your system's `PATH`. Please follow the installation instructions in the Prerequisites section.

2.  **GCS Access Denied / 401 Errors**: This indicates an authentication or permission issue.
    -   Ensure you have registered for the Waymo dataset.
    -   Run `gcloud auth login` and `gcloud auth application-default login` to authenticate.
    -   Make sure your GCP user or service account has `Storage Object Viewer` permissions on the `gs://waymo_open_dataset_v_2_0_1/` bucket.

3.  **Corrupted Files**: If a specific Parquet file fails to process, it might be corrupted. The script is designed to be robust and will log an error and skip the corrupted segment, continuing with the rest of the data.
