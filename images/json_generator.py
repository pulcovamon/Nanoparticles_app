'''
Script for creating .json for scales of microscopy images.
'''
import os
import json

def loading(folder_path, json_path):
    files = {}
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.name[-3:] == ('jpg' or 'png'):
                files[entry.name] = ['', '']

    with open(json_path, 'w') as json_file:
        json.dump(files, json_file)


if __name__ == '__main__':
    loading('/home/monika/Desktop/project/Nanoparticles_app/images',
        '/home/monika/Desktop/project/Nanoparticles_app/images/scales.json')
