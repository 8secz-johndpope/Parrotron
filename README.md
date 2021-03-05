# 아직 제작중입니다.

# Parrotron Speech-to-Speech
|Strategy|Dataset|loss|WER| 
|--------|-----|-------|------|
|E0|Librispeech_test-clean|?||*|
|E1|Librispeech_test-clean|?||?|

* E0 : Parrotron_no_ASR_decoder 
* E1 : Parrotron 

## Parrotron Intro
Pytorch implementation of "Parrotron: An End-to-End Speech-to-Speech Conversion Model and its Applications to Hearing-Impaired Speech and Speech Separation".

## Model
### Feature
* Input wav : 80-dim log-Mel

  parameter | value
  ------|-----
  N_FFT | 1024
  Frame length | 20ms
  Frame shift | 12.5ms
  Window function | hann window
  fmin | 125
  fmin | 7600

* Targets wav : 1025-dim STFT 

  parameter | value
  ------|-----
  FFT | 2048
  Frame length | 20ms
  Frame shift | 12.5ms
  Window function | hann window

### Architecture
<img width = "400" src = "https://user-images.githubusercontent.com/43025347/109615901-d7fdef00-7b77-11eb-8592-95c4ce285ce4.png">

### Print Model
```python

```

## Data
### Dataset information
Train data set : train-clean-100
Test data set : test-clean

LibriSpeech ASR corpus : https://www.openslr.org/12

### Data format
* 음성 데이터 : 16bit, mono 16k sampling WAV audio
* 정답 스크립트 : 
  ```js
    잠시보류
  ```

### Dataset folder structure
* DATASET-ROOT-FOLDER
```
|--DATA
   |--train
      |--wav
         +--a.wav, b.wav, c.wav ...
      |--TTS_wav
         +--a.wav, b.wav, c.wav ...
      |--txt
         +--a.txt, b.txt, c.txt ...
   |--test
      |--wav
         +--a_test.wav, b_test.wav, c_test.wav ...
      |--TTS_wav
         +--a_test.wav, b_test.wav, c_test.wav ...
      |--txt
         +--a_test.txt, b_test.txt, c_test.txt ...
```
* train.csv, test.csv
  ```
  <wav-path>,<script-path>,<TTS_wav-path>
  460-172359-0076.wav,460-172359-0076.txt,460-172359-0076.wav
  460-172359-0077.wav,460-172359-0077.txt,460-172359-0077.wav
  460-172359-0078.wav,460-172359-0078.txt,460-172359-0078.wav
  460-172359-0079.wav,460-172359-0079.txt,460-172359-0079.wav
  460-172359-0080.wav,460-172359-0080.txt,460-172359-0080.wav
  ...
  ```
* english_unit.labels
  ```
  #id\char 
  0   _
  1   A
  2   B
  3   C
  ...
  26   Z
  27   '
  28    
  29   <s>
  30   </s>
  ```

## References
### Git hub References
* https://github.com/NVIDIA/tacotron2
* https://github.com/sooftware

### Paper References
* Parrotron: An End-to-End Speech-to-Speech Conversion Model and its Applications to Hearing-Impaired Speech and Speech Separation (https://arxiv.org/abs/1904.04169)

## computer power
* NVIDIA TITAN Xp * 4

## Contacts
학부생의 귀여운 시도로 봐주시고 해당 작업에 대한 피드백, 문의사항 모두 환영합니다.

fd873630@naver.com로 메일주시면 최대한 빨리 답장드리겠습니다.