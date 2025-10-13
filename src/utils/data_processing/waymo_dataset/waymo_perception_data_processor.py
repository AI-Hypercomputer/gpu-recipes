# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
import os
import subprocess
import threading
from typing import List
import pandas as pd
from io import BytesIO
from datasets import Dataset, Features, Value, Image as ImageFeature
import pyarrow as pa
import pyarrow.parquet as pq
import click

from PIL import Image
import glob
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


class CameraName(enum.IntEnum):
    UNKNOWN = 0
    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5


class LabelType(enum.IntEnum):
    # Anything that does not fit the other classes or is too ambiguous to
    # label.
    TYPE_UNDEFINED = 0
    # The Waymo vehicle.
    TYPE_EGO_VEHICLE = 1
    # Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
    TYPE_CAR = 2
    # Large vehicle that carries cargo.
    TYPE_TRUCK = 3
    # Large vehicle that carries more than 8 passengers.
    TYPE_BUS = 4
    # Large vehicle that is not a truck or a bus.
    TYPE_OTHER_LARGE_VEHICLE = 5
    # Bicycle with no rider.
    TYPE_BICYCLE = 6
    # Motorcycle with no rider.
    TYPE_MOTORCYCLE = 7
    # Trailer attached to another vehicle or horse.
    TYPE_TRAILER = 8
    # Pedestrian. Does not include objects associated with the pedestrian, such
    # as suitcases, strollers or cars.
    TYPE_PEDESTRIAN = 9
    # Bicycle with rider.
    TYPE_CYCLIST = 10
    # Motorcycle with rider.
    TYPE_MOTORCYCLIST = 11
    # Birds, including ones on the ground.
    TYPE_BIRD = 12
    # Animal on the ground such as a dog, cat, cow, etc.
    TYPE_GROUND_ANIMAL = 13
    # Cone or short pole related to construction.
    TYPE_CONSTRUCTION_CONE_POLE = 14
    # Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
    TYPE_POLE = 15
    # Large object carried/pushed/dragged by a pedestrian.
    TYPE_PEDESTRIAN_OBJECT = 16
    # Sign related to traffic, including front and back facing signs.
    TYPE_SIGN = 17
    # The box that contains traffic lights regardless of front or back facing.
    TYPE_TRAFFIC_LIGHT = 18
    # Permanent building and walls, including solid fences.
    TYPE_BUILDING = 19
    # Drivable road with proper markings, including parking lots and gas
    # stations.
    TYPE_ROAD = 20
    # Marking on the road that is parallel to the ego vehicle and defines
    # lanes.
    TYPE_LANE_MARKER = 21
    # All markings on the road other than lane markers.
    TYPE_ROAD_MARKER = 22
    # Paved walkable surface for pedestrians, including curbs.
    TYPE_SIDEWALK = 23
    # Vegetation including tree trunks, tree branches, bushes, tall grasses,
    # flowers and so on.
    TYPE_VEGETATION = 24
    # The sky, including clouds.
    TYPE_SKY = 25
    # Other horizontal surfaces that are drivable or walkable.
    TYPE_GROUND = 26
    # Object that is not permanent in its current position and does not belong
    # to any of the above classes.
    TYPE_DYNAMIC = 27
    # Object that is permanent in its current position and does not belong to
    # any of the above classes.
    TYPE_STATIC = 28


CAMERA_NAME_MAP = {member.value: member.name for member in CameraName}
WAYMO_LABEL_ID_TO_NAME = {member.value: member.name for member in LabelType}

# --- Configuration ---
WAYMO_DATA_ROOT = "gs://waymo_open_dataset_v_2_0_1/training"
LOCAL_DATA_ROOT = "./waymo_data"
THREAD_COUNTS = 4
CAMERA_IMAGE_DIR = os.path.join(LOCAL_DATA_ROOT, "camera_image")
CAMERA_BOX_DIR = os.path.join(LOCAL_DATA_ROOT, "camera_box")


def _download_dataset_locally(input_dir: str):
    """
    Downloads dataset segments from GCS to local storage.
    """
    # Construct remote paths
    remote_camera_image_path = os.path.join(WAYMO_DATA_ROOT, "camera_image")
    remote_camera_box_path = os.path.join(WAYMO_DATA_ROOT, "camera_box")

    local_camera_image_target = os.path.join(input_dir, "camera_image")
    local_camera_box_target = os.path.join(input_dir, "camera_box")

    paths_to_download = [
        (local_camera_image_target, remote_camera_image_path),
        (local_camera_box_target, remote_camera_box_path),
    ]

    for local_path_dir, remote_path_item in paths_to_download:
        os.makedirs(local_path_dir, exist_ok=True)
        # The gsutil command needs to handle both file and directory copy.
        # If remote_path_item ends with '.parquet', it's a file.
        # Otherwise, it's a directory, so we append '/*' to copy its contents.
        # The local_path_dir is always the target directory.
        source_for_gsutil = remote_path_item
        if not remote_path_item.endswith(".parquet"):
            # If PARQUET_ID is empty, download all parquets in the directory
            source_for_gsutil = os.path.join(remote_path_item, "*.parquet")

        gsutil_command = ["gsutil", "-m", "cp", "-r", source_for_gsutil, local_path_dir]

        logger.info(
            f"[DATALOADER] Downloading dataset. Command: {' '.join(gsutil_command)}"
        )
        try:
            result = subprocess.run(
                gsutil_command, capture_output=True, text=True, check=True
            )
            logger.info(f"[DATALOADER] Successfully downloaded to {local_path_dir}.")
            if result.stdout:
                logger.info(f"[DATALOADER] gsutil stdout: {result.stdout}")
            if result.stderr:  # gsutil often prints status to stderr even on success
                logger.info(f"[DATALOADER] gsutil stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"[Fatal][DATALOADER] Failed to download from {source_for_gsutil} to {local_path_dir}. "
                f"Return code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )
            raise Exception(
                f"Failed to download dataset from {source_for_gsutil}: {e.stderr}"
            )
        except Exception as e:
            logger.error(
                f"[Fatal][DATALOADER] An unexpected error occurred during download: {e}"
            )
            raise


def process_waymo_data(camera_image_df, camera_box_df):
    """
    Processes data for a single Waymo segment to extract images,
    and generate questions/answers based on camera boxes.
    Returns a list of dictionaries, each containing an image, question, and answer.
    """
    segment_data = []

    # Group camera boxes by frame timestamp and camera name for easier lookup
    boxes_by_frame_camera = defaultdict(list)
    if camera_box_df is not None:
        for _, row in camera_box_df.iterrows():
            frame_ts = row["key.frame_timestamp_micros"]
            camera_name_int = row["key.camera_name"]
            camera_name_str = CAMERA_NAME_MAP.get(camera_name_int, "UNKNOWN_CAMERA")
            boxes_by_frame_camera[(frame_ts, camera_name_str)].append(row)

    if camera_box_df is not None:  # Only log if box_df was provided
        logger.info(
            f"[DataLoader][Process Waymo data] Grouped {sum(len(v) for v in boxes_by_frame_camera.values())} boxes from {len(boxes_by_frame_camera)} unique (frame_ts, camera) pairs."
        )

    for _, row in camera_image_df.iterrows():
        image_bytes = row["[CameraImageComponent].image"]
        camera_name_int = row["key.camera_name"]
        frame_ts = row["key.frame_timestamp_micros"]

        camera_name_str = CAMERA_NAME_MAP.get(camera_name_int, "UNKNOWN_CAMERA")

        try:
            pil_image = (
                Image.open(BytesIO(image_bytes)).convert("RGB").resize((250, 200))
            )
        except Exception as e:
            logger.warning(
                f"Could not process image for frame_ts {frame_ts}, camera {camera_name_str}. Error: {e}"
            )
            continue  # Skip this image if it's corrupted or unreadable

        # Get labels for this specific frame and camera
        current_frame_boxes = boxes_by_frame_camera.get((frame_ts, camera_name_str), [])

        detected_objects = set()
        for box_row in current_frame_boxes:
            label_id = box_row["[CameraBoxComponent].type"]
            detected_objects.add(WAYMO_LABEL_ID_TO_NAME.get(label_id, "Unknown"))

        if detected_objects:
            question = f"What objects are visible in the {camera_name_str} image at timestamp {frame_ts}?"
            label = "Objects: " + ", ".join(sorted(list(detected_objects))) + "."
        else:
            question = f"Are there any common objects visible in the {camera_name_str} image at timestamp {frame_ts}?"
            label = "No common objects detected."

        segment_data.append(
            {"image": pil_image, "question": question, "multiple_choice_answer": label}
        )

    logger.info(
        f"[DataLoader][Process Waymo data] Processed {len(segment_data)} images for the current segment."
    )
    return segment_data


def load_waymo_data_to_dataset(
    image_parquet_files: List[str],
    box_parquet_files: List[str],
    output_dir: str,
    output_filename_base: str,  # = "waymo_processed_dataset.parquet"
):
    """
    Loads Waymo camera images and annotations from local Parquet files,
    processes them, creates a Hugging Face Dataset, and saves it to a Parquet file locally.
    """

    all_images = []
    all_questions = []
    all_answers = []

    box_dfs_by_segment = {}
    for box_file in box_parquet_files:
        # Extract segment name, assuming format like 'segment_id_string.parquet'
        segment_name = os.path.basename(box_file).replace(".parquet", "")
        logger.info(
            f"[Dataloader] Loading box data from {box_file} for segment {segment_name}."
        )
        try:
            box_dfs_by_segment[segment_name] = pd.read_parquet(
                box_file, engine="pyarrow"
            )
        except Exception as e:
            # Log as error but continue, to allow processing of segments with valid image data
            # if box data is optional or partially available.
            logger.error(
                f"[Dataloader] Could not load or read box file {box_file}: {e}. This segment's boxes will be unavailable."
            )
    logger.info(f"[Dataloader] Loaded {len(box_dfs_by_segment)} box dataframes.")

    loaded_segments_count = 0
    for image_file in image_parquet_files:
        segment_name = os.path.basename(image_file).replace(".parquet", "")
        logger.info(
            f"[Dataloader] Processing segment: {segment_name} from file {image_file}"
        )

        try:
            current_camera_image_df = pd.read_parquet(image_file, engine="pyarrow")
            # Retrieve the corresponding box dataframe; it's okay if it's None (handled in process_waymo_data)
            current_camera_box_df = box_dfs_by_segment.get(segment_name)
            if current_camera_box_df is None:
                logger.warning(
                    f"[Dataloader] No box data found for segment: {segment_name}. Processing with images only."
                )

            segment_processed_data = process_waymo_data(
                current_camera_image_df, current_camera_box_df
            )

            for item in segment_processed_data:
                all_images.append(item["image"])
                all_questions.append(item["question"])
                all_answers.append(item["multiple_choice_answer"])

            if segment_processed_data:  # only increment if data was actually processed
                loaded_segments_count += 1

        except Exception as e:
            logger.error(
                f"[Fatal][Dataloader] Error loading or processing image file {image_file}: {e}. Skipping this segment."
            )
            # Continue to next segment to make it more robust
            # If one file is corrupted, we might still process others.
            # Consider if a fatal error should stop all processing.
            # raise if any single file error should be fatal

    logger.info(
        f"[Dataloader] Successfully processed data for {loaded_segments_count} segments."
    )

    if not all_images:
        logger.warning(
            "[Dataloader] No data was loaded into the lists. Resulting dataset will be empty."
        )

    dataset_dict = {
        "image": all_images,
        "question": all_questions,
        "multiple_choice_answer": all_answers,
    }
    logger.info(
        f"[Dataloader] Creating Hugging Face Dataset from lists with lengths images={len(all_images)},"
        f"questions={len(all_questions)}, answers={len(all_answers)}."
    )

    try:
        # Define the features, especially for the image data, to help serialization
        features = Features(
            {
                "image": ImageFeature(),
                "question": Value("string"),
                "multiple_choice_answer": Value("string"),
            }
        )
        dataset = Dataset.from_dict(dataset_dict, features=features)
        logger.info(
            f"[Dataloader] Created Hugging Face Dataset with {len(dataset)} entries."
        )
    except Exception as e:
        logger.error(f"[Fatal][Dataloader] Failed to create Hugging Face Dataset: {e}")
        logger.error(
            f"Lengths of lists: images={len(all_images)}, questions={len(all_questions)}, answers={len(all_answers)}"
        )
        # You might want to inspect the data here or save it in a different format for debugging
        raise Exception(f"Failed to create dataset: {e}")

    # --- Save dataset to Parquet ---
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Construct output filename, incorporating PARQUET_ID if it exists
    segment_name = os.path.basename(image_parquet_files[0])
    final_output_filename = output_filename_base + "_" + segment_name

    output_parquet_path = os.path.join(output_dir, final_output_filename)

    logger.info(
        f"[Dataloader] Attempting to save dataset to Parquet format at: {output_parquet_path}"
    )
    try:
        dataset.to_parquet(output_parquet_path)
        logger.info(f"[Dataloader] Successfully saved dataset to {output_parquet_path}")
    except Exception as e:
        logger.error(f"[Fatal][Dataloader] Failed to save dataset to Parquet: {e}")
        raise  # Re-raise the exception after logging


def thread_work(
    thread_id: int,
    image_parquet_files: List[str],
    box_parquet_files: List[str],
    output_dir: str,
    output_filename_base: str,
) -> None:
    """Thread worker function to process a subset of Waymo data."""
    for i, (image_file, box_file) in enumerate(
        zip(sorted(image_parquet_files), sorted(box_parquet_files))
    ):
        logger.info(
            f"[Thread {thread_id}] Processing image file: {image_file}, box file: {box_file}"
        )
        load_waymo_data_to_dataset(
            image_parquet_files=[image_file],
            box_parquet_files=[box_file],
            output_dir=output_dir,
            output_filename_base=output_filename_base,
        )


@click.command()
@click.option(
    "-o",
    "--output_dir",
    type=str,
    default="./waymo_processed_dataset",
    help="Output directory to save the processed dataset.",
)
@click.option(
    "-i",
    "--input_dir",
    type=str,
    default="./waymo_data",
    help="Directory containing the camera images.",
)
@click.option(
    "-n",
    "--num_threads",
    type=int,
    default=THREAD_COUNTS,
    help="Number of threads to use for processing.",
)
@click.option(
    "--output_filename_base",
    type=str,
    default="waymo_processed_dataset",
    help="Base filename for the output Parquet dataset.",
)
@click.option(
    "-d",
    "--download",
    is_flag=True,
    help="Download the dataset locally.",
)
def main(
    output_dir: str,
    input_dir: str,
    num_threads: int,
    output_filename_base: str,
    download: bool,
):
    """Waymo open source perception dataset preprocessing script.

    This function orchestrates the data processing pipeline. It performs the
    following steps:
    1. Parses command-line arguments for input/output directories, threading,
       and download options.
    2. Optionally downloads the Waymo dataset from GCS if the --download flag
       is specified.
    3. Discovers all image and bounding box Parquet files in the input
       directory.
    4. Divides the list of files among a specified number of threads.
    5. Spawns a thread for each subset of files, which calls the `thread_work`
       function to process the data.
    6. Waits for all threads to complete their execution.
    """

    logger.info("[Dataloader] Starting dataset download locally (if needed).")
    if download:
        _download_dataset_locally(input_dir)
    logger.info("[Dataloader] Dataset download finished.")

    image_dir = os.path.join(input_dir, "camera_image")
    box_dir = os.path.join(input_dir, "camera_box")
    thread_counts = num_threads

    image_parquet_files = sorted((glob.glob(os.path.join(image_dir, "*.parquet"))))
    box_parquet_files = sorted((glob.glob(os.path.join(box_dir, "*.parquet"))))

    if not image_parquet_files:
        logger.warning(
            f"[Dataloader] No image Parquet files found in {image_dir}. Ensure data was downloaded correctly."
        )

    logger.info(f"[Dataloader] Found {len(image_parquet_files)} image Parquet files.")
    logger.info(f"[Dataloader] Found {len(box_parquet_files)} box Parquet files.")

    threads = []
    files_per_thread = math.ceil(len(image_parquet_files) / thread_counts)

    for i in range(thread_counts):
        start_index = i * files_per_thread
        end_index = min((i + 1) * files_per_thread, len(image_parquet_files))

        sub_image_parquet_files = image_parquet_files[start_index:end_index]
        sub_box_parquet_files = box_parquet_files[start_index:end_index]
        logger.info(
            f"[Dataloader] Thread - {i}: Processing image files {sub_image_parquet_files} and box files {sub_box_parquet_files}"
        )
        thread = threading.Thread(
            target=thread_work,
            args=(
                i,
                sub_image_parquet_files,
                sub_box_parquet_files,
                output_dir,
                output_filename_base,
            ),
            name=f"worker-{i}",
        )
        threads.append(thread)
        thread.start()
        logger.info(f"[Dataloader] Started thread {thread.name}")

    for thread in threads:
        thread.join()

    logger.info("[Dataloader] All threads completed.")


if __name__ == "__main__":
    main()
