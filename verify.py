import os
import numpy as np
import sys

def decode(array_to_decode, key):
    ret = key ^ array_to_decode
    return ret

dst_folder = sys.argv[1]
key_code_string = sys.argv[2]
key_code = int('0x' + key_code_string, 16)


show_option = False
if len(sys.argv) == 4:
    if sys.argv[3] == "yes":
        show_option = True

key8 = np.array(key_code, np.uint8)

file_count = 0
pass_count = 0
fail_count = 0
print("\nVerifying ...")

with open(dst_folder + 'matching.txt', 'rt', encoding='utf8') as fh_mtch:
    for line in fh_mtch:
        file_count += 1
        file_and_path = line.strip().split('>')

        src_path_and_file = file_and_path[1] + "\\" + file_and_path[0]

        encoded_fname = dst_folder + file_and_path[0]
        fp = open(encoded_fname, "rb")
        bt_obj1 = fp.read()
        fp.close()

        bt_obj1_np = np.array(list(bt_obj1), np.uint8)

        decoded_data = decode(bt_obj1_np, key8)

        fp1 = open(src_path_and_file, "rb")
        bt_obj2 = fp1.read()
        fp1.close()

        bt_obj2_np = np.array(list(bt_obj2), np.uint8)
        ret = bytes(decoded_data) == bt_obj2

        if ret == True:
            if show_option == True:
                print("#{0:03d}: {1}({2:,}), pass!!".format(pass_count, line.split('>')[0], os.path.getsize(src_path_and_file)))
                print("location: {0}".format(line.split('>')[1]))

            pass_count += 1
        else:
            print("#{0:03d}: {1}({2:,}), fail ********".format(file_count-pass_count, line.split('>')[0],
                                                  os.path.getsize(src_path_and_file)))
            print("location: {0}".format(line.split('>')[1]))

print(f"\n### total {file_count} files: Pass={pass_count}, Fail={file_count-pass_count}")
