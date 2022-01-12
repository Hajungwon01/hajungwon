"""

개요
    레포트 2의 암호화/복호화/검증을 수행하는 메인 프로그램
    평가의 수월성을 위해 암호화 프로그램을 파이썬에서 수행한다.
    encode, verify 단계에 소요되는 시간을 측정하여 각 단계와 총합 시간을 출력한다.

주의 사항
    다른 템플레이트 파일과는 달리 본 main.py는 수정하지 말고 그대로 제출해 주세요.
    지시한 대로 제출하지 않으면 원본과 교체해서 평가할 예정입니다.

"""
import os
import numpy as np
import time

# 주요 변수명 - 아래 선언 변수 이름과 그 값들은 그대로 사용해 주세요. 주석문은 지워도 됩니다.
# 실제 폴더의 위치나, 확장자, 키코드 스트링의 값은 채점 당시 평가자가 바꿀 수 있습니다.
src_folder = 'd:\\python2021\\src\\'  # 검색대상 폴더 스트링
dst_folder = 'd:\\python2021\\dst\\'  # 암호화하여 저장할 폴더 스트링

target_file1 = 'hwp_pptx'            # @@ 평가 1: 2종류, 확장자명은 바뀔 수 있음
target_file2 = 'jpg_png_hwp_pptx'   # @@ 평가 2: 4종류, 확장자명은 바뀔 수 있음
target_file3 = 'jpg_pptx_txt_py_hwp_zip_png'  # @@ 평가 3:6종류 이상의 파일이 암호화 작업의 대상

# 테스트할 검색 대상 확장자 옵션을 선택한다.
target_file = target_file3     # target_file1, target_file2, target_file3 중의 하나.

# 대상 파일의 종류에 제한받지 않게 설계되었는지를 추가로 평가합니다.
key_code_str = '3c'   # 16진수이며 스트링임에 유의!!! 암호화에 사용할 8비트 16진수 코드값 스트링

time1 = time.time()

# ------------------------------------------------------------------------------------------------
# encoding을 시행한다.
# python encode.py <검색대상 폴더 위치> <파일 저장할 폴더 위치> <선택할 파일의 확장자> <key_code_in_hex>
# ------------------------------------------------------------------------------------------------
os.system(f"python encode.py {src_folder} {dst_folder} {target_file} {key_code_str}")
time2 = time.time()


# ------------------------------------------------------------------------------------------------
# verifying을 시행한다.
# python verify.py <암호화 파일이 저장된 폴더 위치> <key_code_str> <show_option>
# ------------------------------------------------------------------------------------------------
# 다음 3가지 동작이 올바르게 동작하는지 점검합니다. 제출할 떄는 1번으로 고정하세요.
#os.system(f"python verify.py {dst_folder} {key_code_str} yes")      # 1) 파일별 비교 결과를 출력한다.
#os.system(f"python verify.py {dst_folder} {key_code_str} no")       # 2) 파일별 비교 결과를 출력하지 않는다.
os.system(f"python verify.py {dst_folder} {key_code_str}")         # 3) 파일별 비교 결과를 출력하지 않는다.


time3 = time.time()

print(f"\nEncode Execution Time = {time2-time1:.4f}[sec.]")
print(f"Verify Execution Time = {time3-time2:.4f}[sec.]")
print(f"Total Execution Time = {time3-time1:.4f}[sec.]")

# 프로그램별 용량을 출력한다.
sz1 = os.path.getsize('encode.py')
print(f'\nsize of encode.py={sz1:#,}[bytes]')

sz2 = os.path.getsize('verify.py')
print(f'size of verify.py={sz2:,}[bytes]')


print(f"Sum of 2 Programs={sz1+sz2:,}[bytes]")

