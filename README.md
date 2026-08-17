# In-Context Source and Channel Coding

This repository implements an ICD-aided SSCC text transmission pipeline. Following the paper, the transmitter remains a standard LLM-based arithmetic source coder, while the receiver is enhanced with ICD to mitigate the cliff effect at low SNR.

Main flow:

```
compress.py
  -> ECCT_forward_v1_final.py
  -> extract_message.py
  -> decompress_v2_sampling.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. LLM-AC Source Compression

```
python compress.py
```

`compress.py` tokenizes the text and compresses it into a binary bitstream using arithmetic coding driven by LLM token probabilities.

### 2. LDPC Transmission and ECCT Decoding

```
python ECCT_forward_v1_final.py 
```

`ECCT_forward_v1_final.py` applies systematic LDPC coding, BPSK transmission over AWGN/Rayleigh channels, and ECCT decoding. It outputs both decoded codewords and ECCT-derived bit probabilities/reliability.


### 3. Information-Bit Extraction

```
python extract_message.py
```

Because the LDPC code is systematic, `extract_message.py` recovers the estimated source bitstream by taking the first `K` information bits from each ECCT-decoded codeword.

### 4. ICD-Based Source Decoding

```
python decompress_v2_sampling.py
```

`decompress_v2_sampling.py` implements the receiver-side ICD idea: it builds reliability-guided candidate bitstreams, samples a compact and diverse candidate subset, decodes candidates with the LLM arithmetic decoder, and selects the final text by combining ECCT-side reliability with LLM log-likelihood.

# Paper

Ziqiong Wang, Tianqi Ren, **Rongpeng Li**, Zhifeng Zhao, and Honggang Zhang, “In-context source and channel coding,” *SCIENCE CHINA Information Sciences*, Aug. 2026.