import numpy as np
import os
from tqdm import tqdm
from utils import set_seed
import warnings
warnings.filterwarnings("ignore")


def read_message_from_txt(filename):
    message_list = []
    with open(filename, 'r') as f:
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
            message = [int(x) for x in message_string]
            message_list.append(message)
    return message_list

def decode_bitstream(bitstream, output_filename, N, K):
    decoded_bitstream = ""
    for i in range(0, len(bitstream), N):
        segment = bitstream[i : i+N]
        
        x = np.array([int(bit) for bit in segment])
        m = np.copy(x[:K]) # 伪LDPC译码，直接取了前K位作为消息
        
        decoded_segment = "".join(map(str, m))
        decoded_bitstream += decoded_segment
    
    with open(output_filename, 'a') as f:
        f.write(decoded_bitstream+'\n')
    # print(f"Message extraction completed, results saved to'{output_filename}'。")


seed = 42
set_seed(seed)


root_dir_in = 'mydata/ECCT_pred'
root_dir_out = 'mydata/ECCT_extracted'
filename = 'demo_decode'

N, K = 49, 24
code_name = f'LDPC_K{K}_N{N}'
channel = 'AWGN'  # AWGN or Rayleigh

for snr in tqdm(range(-3, 16, 3)):
    input_file = os.path.join(root_dir_in, code_name, channel, filename+'_SNR_'+str(snr)+'.txt')
    output_file = os.path.join(root_dir_out, code_name,  channel, filename+'_SNR_'+str(snr)+'.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    codeword_list = read_message_from_txt(input_file)
    for input_bitstream in tqdm(codeword_list):
        decode_bitstream(input_bitstream, output_file, N, K)