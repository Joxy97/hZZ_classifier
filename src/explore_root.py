import numpy as np
import uproot
import sys
import os
import argparse

def main(root_file):

    running = True

    with uproot.open(root_file) as f:
        print(f"\nAnalyzing file {os.path.basename(root_file)}:\n")
        keys = f.keys()
        current = ''   
        print("\nroot_file" + current)
        print(f.keys())
         
        while running == True:    
            choice = input()
            if choice in keys:
                current = current + '/' + choice
                if f[current].keys() == []:
                    print("\nroot_file" + current)
                    print(f[current].array())
                else:
                    keys = f[current].keys()
                    print("\nroot_file" + current)
                    print(keys)
            if choice == 'exit':
                running = False
                sys.exit()
            if choice == 'back':
                current = current.rsplit('/', 1)[0]
                if current == '':
                    keys = f.keys()
                else:
                    keys = f[current].keys()
                print("\nroot_file" + current)
                print(keys)
            if choice == 'pwd':
                print(current.lstrip('/'))
            if choice == 'len':
                if f[current].keys() == []:
                    print(len(f[current].array()))
                else:
                    keys = f[current].keys()
                    print(len(f[current]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ROOT files to HDF5 format with train/test split")
    parser.add_argument('root_file', type=str, help='Folder containing ROOT files')

    args = parser.parse_args()

    main(args.root_file)
