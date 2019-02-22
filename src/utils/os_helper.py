import os


"""
Contains all os related helper functions e.g.
File size
TODO:
File name generation
File path generation
Cmd line arugment or config parser auto save
"""

def get_num_data_points(file_path, size_of_one_data_point_bytes):
    # calculates number of data points in a memmap file, works for any file
    file_size_bytes = os.path.getsize(file_path)
    return int(file_size_bytes / size_of_one_data_point_bytes)
