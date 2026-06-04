# EU.jsonl 已過濾，只跑 language=en（約 200 題）
# 用 --layer_idx 指定 Probing 後的 layer
# 測資位置：data/EU.jsonl（EmoBench EU split）
# model 已改為 Hugging Face：google/gemma-2-9b-it（需 HF_TOKEN 並接受 Gemma 授權）
# 檢查答案的方法是看answer有沒有在response裡面出現，因此有可能出錯

Baseline:
acc so far: 94 / 200
acc 94 all_num 200 token_num 138.775

layer 31, 1810
acc so far: 101 / 200
acc 101 all_num 200 token_num 142.18

layer 31, 13375
acc so far: 90 / 200
acc 90 all_num 200 token_num 136.135

layer 31, 12813
acc so far: 92 / 200
acc 92 all_num 200 token_num 140.38

layer 9, 4409
acc so far: 95 / 200
acc 95 all_num 200 token_num 140.435

layer 9, 5800
acc so far: 91 / 200
acc 91 all_num 200 token_num 142.555

layer 9, 2498
acc so far: 97 / 200
acc 97 all_num 200 token_num 140.98

layer 20, 12346
acc so far: 92 / 200
acc 92 all_num 200 token_num 141.67

layer 20, 7560
acc so far: 93 / 200
acc 93 all_num 200 token_num 139.11

layer 20, 3967
acc so far: 96 / 200
acc 96 all_num 200 token_num 143.63

# Scripts

```shell
python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 31 --feature_idx 3879 319 8831 1088 10538 3003 9189 14062 1810 8058 --strength 1.0 --dataset LLM

python sae_gsm.py --model_name gemma-2-9b-it --dataset LLM

# llm feature
python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 31 --feature_idx 3879 319 8831 1088 10538 3003 9189 14062 1810 8058 --strength 1.0 --dataset EU

# twitter feature
python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 31 --feature_idx 1810 13375 12813 2027 8011 4044 1662 14062 1088 16127 --strength 1.0 --dataset EU

python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 20 --feature_idx 12346 7560 3967 973 3638 14555 11180 16067 10418 5245 --strength 1.0 --dataset EU

python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 9 --feature_idx 1735 6353 248 472 7441 6386 15509 2498 5800 4409 --strength 1.0 --dataset EU

python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 5 --feature_idx 16377 16378 16379 16380 16381 16382 16383 14590 12808 9445 --strength 1.0 --dataset EU

python sae_gsm.py --model_name gemma-2-9b-it --dataset EU
```

# New Results

| dataset | steering layer | acc     | acc (complex emotions) | acc (personal beliefs) | acc (emotion cues) | acc (perspective taking) |
| ------- | -------------- | ------- | ------- | ------- | ------- | ------- |
| LLM     | 31             | 218/250 |         |         |         |         |
| LLM     | None           | 217/250 |         |         |         |         |
| EU      | 9              | 95/200  | 28/49   | 26/56   | 15/28   | 26/67   |
| EU      | 31             | 88/200  | 27/49   | 20/56   | 14/28   | 27/67   |
| EU      | None           | 86/200  | 27/49   | 19/56   | 15/28   | 25/67   |
