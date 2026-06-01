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



python sae_gsm.py --model_name gemma-2-9b-it --steering \
  --layer_idx 31 --feature_idx 12813 --strength 0.5
