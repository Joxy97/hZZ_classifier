import os
import h5py
import numpy as np
import uproot
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_datasets(events, variables_to_extract):
    """
    Extract the variables of interest from the ROOT events.
    """
    datasets = {}
    for group, variables in variables_to_extract.items():
        for var in variables:
            try:
                data = events[var].to_numpy()
                datasets[f"INPUTS/{group}/{var}"] = data
            except KeyError:
                print(f"Warning: Variable {var} not found in events.", flush=True)
    return datasets

def get_label_from_filename(file_name, label_mappings):
    """
    Assign a label based on the filename using a flexible mapping.
    :param file_name: The name of the ROOT file.
    :param label_mappings: A dictionary where keys are substrings and values are the labels.
    :return: The assigned label based on the mapping.
    """
    for substring, label in label_mappings.items():
        if substring in file_name:
            return label
    return 0  # Default label if no substring matches

def process_file_chunk(file_name, entry_start, entry_stop, variables_to_extract, label):
    """
    Process a chunk of the ROOT file and extract the required datasets, along with labels.
    """
    with uproot.open(file_name) as in_file:
        events = in_file["ZZTree/candTree;1"].arrays(entry_start=entry_start, entry_stop=entry_stop)
        datasets = get_datasets(events, variables_to_extract)
        num_events = len(events[variables_to_extract['Leptons'][0]])  # Number of events in this chunk (from Leptons)
        labels = np.full((num_events,), label)  # Create an array of labels for this chunk
        datasets['LABELS'] = labels
    return datasets

def main(input_folder, output_file, train_frac, load_size, num_workers, label_mappings):
    variables_to_extract = {
        'Leptons': ['LepPt', 'LepEta', 'LepPhi'],
        'Z1': ['Z1Mass', 'Z1Pt'],
        'Z2': ['Z2Mass', 'Z2Pt'],
        'ZZ': ['ZZMass', 'ZZEta', 'ZZPt']
    }

    all_datasets = {}
    all_labels = []
    root_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.root')]

    print(f"Processing files from folder: {input_folder}", flush=True)
    print(f"Output file: {output_file}", flush=True)
    print(f"Using {num_workers} parallel workers.", flush=True)

    total_events_to_process = 0
    processed_events = 0

    # Use ProcessPoolExecutor for parallel processing of files and chunks
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for file_name in root_files:
            label = get_label_from_filename(file_name, label_mappings)  # Get label based on file name and mappings

            with uproot.open(file_name) as in_file:
                num_entries = in_file["ZZTree/candTree;1"].num_entries
                total_events_to_process += num_entries * (1 - train_frac)  # Add to total testing events count

                if "training" in output_file:
                    first_entry = 0
                    final_entry = int(train_frac * num_entries)
                else:
                    first_entry = int(train_frac * num_entries)
                    final_entry = num_entries

                entry_start = first_entry
                entry_stop = None

                # Submit chunks for parallel processing
                while entry_stop != final_entry:
                    entry_stop = min(entry_start + load_size, final_entry)
                    futures.append(
                        executor.submit(process_file_chunk, file_name, entry_start, entry_stop, variables_to_extract, label)
                    )
                    entry_start = entry_stop

        # Collect the results from the parallel processing
        for future in as_completed(futures):
            result = future.result()
            for dataset_name, data in result.items():
                if dataset_name == 'LABELS':
                    all_labels.append(data)
                else:
                    if dataset_name not in all_datasets:
                        all_datasets[dataset_name] = []
                    all_datasets[dataset_name].append(data)

            processed_events += len(data)
            print(f"Processed {processed_events} out of {int(total_events_to_process)} events", flush=True)

    # Combine and save datasets to HDF5
    print("\nSaving data to HDF5...", flush=True)
    with h5py.File(output_file, "w") as output:
        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            output.create_dataset(dataset_name, data=concat_data)
        # Combine and save labels
        concat_labels = np.concatenate(all_labels, axis=0)
        output.create_group('LABELS').create_dataset('labels', data=concat_labels)  # Save labels in LABELS group

    print(f"Data successfully saved to {output_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ROOT files to HDF5 format with train/test split, parallel processing, and flexible label assignment based on file names")
    parser.add_argument('input_folder', type=str, help='Folder containing ROOT files')
    parser.add_argument('output_file', type=str, default="output.h5", help="HDF5 output file")
    parser.add_argument('--train_frac', type=float, default=0.95, help="Fraction of events for training")
    parser.add_argument('--load_size', type=int, default=100000, help="Number of events to load at once")
    parser.add_argument('--num_workers', type=int, default=28, help="Number of parallel workers")

    args = parser.parse_args()

    # Define the label mappings here: substrings in file names mapped to corresponding labels
    label_mappings = {
        '128': 1,   # Files containing '128' get label 1
        # Add more mappings here if needed
    }

    main(args.input_folder, args.output_file, args.train_frac, args.load_size, args.num_workers, label_mappings)