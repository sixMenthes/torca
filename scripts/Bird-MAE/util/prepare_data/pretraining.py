import pandas as pd
from collections import Counter
import random
import argparse

from datasets import load_dataset, Audio, DatasetDict
from birdset.datamodule.components.event_mapping import XCEventMapping
from tqdm import tqdm

def smart_sampling( dataset, label_name, class_limit, event_limit):
    # from BirdSet
    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name), desc="unique_id"
    )
    df = pd.DataFrame(dataset)
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="smart sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        most = current["size"].max()

        while total > class_limit or most != event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                total - (to_del * largest_count) < class_limit
                or most == event_limit
                or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                    path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()

    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)

# pretraining data 
def process_pretraining_dataset(
    cache_dir: "str" = ".",
    dataset_name: "str" = "XCL",       
    num_proc: "int" = 1,           
    revision: "str" = "main",
    mapping_batch_size: "int" = 300,
    mapping_num_proc: "int" = 3,  
    save_path: "str" = ".",          
    class_limit: "int" = 500, 
    event_limit: "int" = 2, 
    hf_path: str = "DBD-research-group/BirdSet", 
    audio_sampling_rate: int = 32_000,
    smart_sampling_label_name: str = "ebird_code", 
    final_columns: list = None 
):
    """
    Loads a dataset from Hugging Face, processes its 'train' split (including audio casting,
    event mapping, optional smart sampling, and column selection), and saves it to disk.

    Args:
        cache_dir: Directory for caching Hugging Face datasets.
        dataset_name: The specific configuration or subset of the dataset to load (e.g., "XCL").
        num_proc: Number of processes for `load_dataset`.
        revision: Specific dataset version (e.g., commit hash) to load.
        mapping_batch_size: Batch size for the event mapping step.
        mapping_num_proc: Number of processes for the event mapping step.
        save_path: The full directory path where the processed dataset should be saved.
                   The naming of this path should reflect whether smart sampling was applied
                   (e.g., include class/event limits or 'allevents'), as this function
                   does not modify `save_path` based on sampling parameters.
        class_limit: If provided (not None) along with `event_limit`, smart sampling is applied
                     with this class limit.
        event_limit: If provided (not None) along with `class_limit`, smart sampling is applied
                     with this event limit.
        hf_path: The main path or name of the dataset on Hugging Face Hub.
        audio_sampling_rate: The sampling rate to which audio data will be cast.
        smart_sampling_label_name: The name of the label column used for smart sampling.
        final_columns: A list of column names to retain in the processed dataset. If None,
                       a default list from the original script is used.

    Returns:
        datasets.DatasetDict: The processed dataset, typically containing the 'train' split.

    Raises:
        ValueError: If the loaded dataset does not contain a 'train' split.

    Note:
        This function assumes that `XCEventMapping` class and `smart_sampling` function
        are defined and available in the scope.
    """
    if final_columns is None:
        final_columns = ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]

    print(f"Loading dataset: {hf_path} (configuration: {dataset_name}, revision: {revision})", flush=True)

    ds = load_dataset(
        path=hf_path,
        name=dataset_name,
        cache_dir=cache_dir,
        num_proc=num_proc,
        revision=revision
    )


    if "train" not in ds:
        raise ValueError(f"Dataset {hf_path} (config: {dataset_name}) loaded successfully, but does not contain a 'train' split.")
    train_data = ds["train"]

    print(f"Casting 'audio' column for 'train' split to {audio_sampling_rate} Hz, mono, decode=False.", flush=True)
    train_data = train_data.cast_column(
        column="audio",
        feature=Audio(
            sampling_rate=audio_sampling_rate,
            mono=True,
            decode=False, 
        ),
    )

    mapper = XCEventMapping()

    print(f"Performing event mapping on 'train' split (batch size: {mapping_batch_size}, num_proc: {mapping_num_proc}).", flush=True)
    train_data = train_data.map(
        mapper, 
        remove_columns=["audio"], 
        batched=True,
        batch_size=mapping_batch_size,
        num_proc=mapping_num_proc,
        desc=f"Event mapping for {dataset_name} train split"
    )

    if class_limit is not None or event_limit is not None:
        print(f"Applying smart sampling to 'train' split: class_limit={class_limit}, event_limit={event_limit}, label='{smart_sampling_label_name}'", flush=True)
        train_data = smart_sampling(
           dataset=train_data,
           label_name=smart_sampling_label_name,
           class_limit=class_limit,
           event_limit=event_limit,
        )
    else:
        print("Skipping smart sampling for 'train' split.", flush=True)

    print(f"Selecting final columns for 'train' split: {final_columns}", flush=True)
    train_data = train_data.select_columns(columns=final_columns)


    processed_ds_dict = DatasetDict({
        "train": train_data
    })

    print(f"Saving processed dataset to: {save_path}", flush=True)
    processed_ds_dict.save_to_disk(save_path)
    print(f"Dataset successfully saved to {save_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save a Hugging Face dataset for pretraining.")

    parser.add_argument("--cache_dir", type=str, default=".", help="Directory for caching Hugging Face datasets.")
    parser.add_argument("--dataset_name", type=str, default="XCL", help="The specific configuration or subset of the dataset to load (e.g., 'XCL').")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes for `load_dataset`.")
    parser.add_argument("--revision", type=str, default="main", help="Specific dataset version (e.g., commit hash) to load.")
    parser.add_argument("--mapping_batch_size", type=int, default=300, help="Batch size for the event mapping step.")
    parser.add_argument("--mapping_num_proc", type=int, default=3, help="Number of processes for the event mapping step.")
    parser.add_argument("--save_path", type=str, required=True, help="Full directory path where the processed dataset should be saved. Name should reflect sampling parameters.")
    parser.add_argument("--hf_path", type=str, default="DBD-research-group/BirdSet", help="The main path or name of the dataset on Hugging Face Hub.")
    parser.add_argument("--audio_sampling_rate", type=int, default=32_000, help="The sampling rate to which audio data will be cast.")
    parser.add_argument("--smart_sampling_label_name", type=str, default="ebird_code", help="The name of the label column used for smart sampling.")
    parser.add_argument("--final_columns", nargs='*', default=None, help="List of column names to retain (e.g., --final_columns col1 col2). If not provided, defaults are used.")
    
    # Smart sampling arguments
    parser.add_argument("--class_limit", type=int, default=500, help="Class limit for smart sampling. Used if --disable_smart_sampling is not set.")
    parser.add_argument("--event_limit", type=int, default=2, help="Event limit for smart sampling. Used if --disable_smart_sampling is not set.")

    args = parser.parse_args()

    # Prepare arguments for the function call
    func_args = vars(args)
    process_pretraining_dataset(**func_args)

