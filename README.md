# Paper

This repository corresponds to the following paper. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

> Ziqiong Wang, Tianqi Ren, Rongpeng Li, Zhifeng Zhao, and Honggang Zhang, “In-context source and channel coding,” *SCIENCE CHINA Information Sciences*, Aug. 2026.
>
> Abstract: Separate Source–Channel Coding (SSCC) remains attractive for text transmission due to its modularity and compatibility with mature entropy coders and powerful channel codes. However, SSCC often suffers from a pronounced cliff effect in low Signal-to-Noise Ratio (SNR) regimes, where residual bit errors after channel decoding can catastrophically break lossless source decoding, especially for Arithmetic Coding (AC) driven by Large Language Models (LLMs). This paper proposes a receiver-side In-Context Decoding (ICD) framework that enhances SSCC robustness without modifying the transmitter. ICD leverages an Error Correction Code Transformer (ECCT) to obtain bit-wise reliability for the decoded information bits. Based on the context-consistent bitstream, ICD constructs a confidence-ranked candidate pool via reliability-guided bit flipping, samples a compact yet diverse subset of candidates, and applies an LLM-based arithmetic decoder to obtain both reconstructions and sequence-level log-likelihoods. A reliability–likelihood fusion rule then selects the final output. We further provide theoretical guarantees on the stability and convergence of the proposed sampling procedure. Extensive experiments over Additive White Gaussian Noise (AWGN) and Rayleigh fading channels demonstrate consistent gains compared with conventional SSCC baselines and representative Joint Source-Channel Coding (JSCC) schemes.

Note that this is a research project and, by definition, is unstable. Please write to us if you find something not correct or strange. 
# In-Context Source and Channel Coding
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
