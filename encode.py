import os
import numpy as np
import sys
def encode(array_to_encode, key):
    data = key ^ array_to_encode
    return data

src_folder = sys.argv[1]
dst_folder = sys.argv[2]
target_file = sys.argv[3]
key_code_string = sys.argv[4]

target_lst = target_file.split('_')

key_code = int('0x' + key_code_string, 16)

key8 = np.array(key_code, np.uint8)

if not os.path.exists(dst_folder): os.makedirs(dst_folder)
fh_mtch = open(dst_folder + 'matching.txt', 'wt', encoding='utf8')

print("\nEncoding....")
mtch_count = 0

for path, sub_dir, files in os.walk(src_folder):
    for filename in files:
        ext = os.path.splitext(filename)[-1][1:]
        if ext in target_lst:
            fh_mtch.write(filename + ">" + path + "\n")
            mtch_count = mtch_count + 1

            full_path_file_name = os.path.join(path, filename)

            bytes_object = open(full_path_file_name, "rb").read()

            bytes_object_np = np.array(list(bytes_object), np.uint8)

            fh_bin = open(dst_folder + filename, 'wb')
            fh_bin.write(encode(bytes_object_np, key8))

fh_mtch.close()

print("mtch_count=", mtch_count)
