import os
import shutil

def copy_directories_structure(src, dst):
    for dirpath, dirnames, filenames in os.walk(src):

        relative_path = os.path.relpath(dirpath, src)
        destination_path = os.path.join(dst, relative_path)

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

src_directory = '/tests/data/data_original'
dst_directory = 'D:\\AGH-Infa\\Sano\\dynamo_library\\tests\\data\\data_tests'

copy_directories_structure(src_directory, dst_directory)
