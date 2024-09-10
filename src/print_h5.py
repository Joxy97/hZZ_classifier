import h5py
import argparse

def print_h5_tree(group, indent=0):
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print("  " * indent + f"Group: {name}")
            print_h5_tree(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * indent + f"Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")

def main(print_file):

    with h5py.File(print_file, 'r+') as file:
        print("")
        print(f'Printing: {print_file.split("/")[-1]}')
        print("")
        print_h5_tree(file)
        print("")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('print_file', type=str, help='Path to the .h5 file')

    args = parser.parse_args()

    main(args.print_file)