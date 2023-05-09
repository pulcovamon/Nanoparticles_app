
'''
Script for creating .json for scales of microscopy images.
'''
import os
import json
import argparse

def parse_command_line():
    """Parse command line arguments.

    Returns:
        folder_path (str): Path to folder with images
        np_type (str): Type of NPs
    """
    parser = argparse.ArgumentParser(
        description="segmentation of NPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        default='data',
        help="Path to folder with images"
    )

    parser.add_argument(
        "--np_type",
        type=str,
        default='nanoparticles',
        help="Type of NPs"
    )

    args = parser.parse_args()
    np_type = args.np_type
    folder_path = args.folder_path

    return folder_path, np_type


def loading(folder_path, json_name, np_type, identificator):
    """Function for creating .json for scales of microscopy images.
    It is necessary to fill scales in .json file.

    Args:
        folder_path (str): Path to folder with images
        json_path (str): Path to .json file
        np_type (str): Type of NPs
        identificator (str): Identificator of the sample
    """
    files = {}
    files['np_type'] = np_type
    files['identificator'] = identificator
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.name[-3:] == ('jpg' or 'png'):
                files[entry.name] = ['', os.path.join(folder_path, entry.name)]
    
    json_path = os.path.join(folder_path, json_name)

    with open(json_path, 'w') as json_file:
        json.dump(files, json_file, indent=4)


if __name__ == '__main__':
    folder_path, np_type = parse_command_line()
    identificator = folder_path.split('/')[-1]
    json_name = f'{identificator}_metadata.json'
    loading(folder_path, json_name, np_type, identificator)