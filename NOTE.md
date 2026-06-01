# 原本 EU dataset 有中文資料，這邊沒有過濾，之後要過濾
# 用 --layer_idx 指定 Probing 後的 layer
# 測資位置：data/EU.jsonl（EmoBench EU split）
# model 已改為 Hugging Face：google/gemma-2-9b-it（需 HF_TOKEN 並接受 Gemma 授權）
# 檢查答案的方法是看answer有沒有在response裡面出現，因此有可能出錯