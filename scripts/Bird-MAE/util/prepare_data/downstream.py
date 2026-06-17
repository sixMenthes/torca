# downstream data  complete
from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_dataset 
import argparse 

def process_downstream_datasets(dataset_names: list[str], cache_dir_base: str):
    """
    Loads and prepares BirdSet datasets.

    Args:
        dataset_names: A list of dataset names to process (e.g., ["PER", "NES"]).
        cache_dir_base: The base directory for caching datasets (e.g., "/scratch/birdset").
    """
    for name in dataset_names:
        print(f"Loading {name}", flush=True)
        cache_path = f"{cache_dir_base}/{name}"
        # Ensure cache directory exists or is created by load_dataset
        dataset = load_dataset("DBD-research-group/BirdSet", name, num_proc=5, cache_dir=cache_path)
        print(f"Loaded {name}", flush=True)

    for name in dataset_names:
        print(f"preparing {name}", flush=True)
        dm = BirdSetDataModule(
            dataset= DatasetConfig(
                data_dir=f"{cache_dir_base}/{name}", 
                hf_path='DBD-research-group/BirdSet',
                hf_name=name,
                n_workers=3,
                val_split=0.0001,
                task="multilabel",
                classlimit=500, 
                eventlimit=5, 
                sampling_rate=32_000,
            ),
        )
        dm.prepare_data()
        print(f"Prepared data for {name} saved to: {dm.disk_save_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and prepare BirdSet datasets.")
    parser.add_argument(
        "--dataset-names",
        nargs='+',
        default=["PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE"],
        help="List of dataset names to process (e.g., PER NES XCL)."
    )
    parser.add_argument(
        "--cache-dir-base",
        type=str,
        default="/data/birdset",
        help="Base directory for caching datasets."
    )

    args = parser.parse_args()
    
    process_downstream_datasets(args.dataset_names, args.cache_dir_base)

