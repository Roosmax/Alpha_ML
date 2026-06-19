# Задача 1 — Кредитный скоринг

Метрика ROC-AUC. Итог: 0.7882

# Запуск
```bash
# LightGBM
python credit_scoring_pipeline.py --seed 42   --tag s42
python credit_scoring_pipeline.py --seed 7    --tag s7
python credit_scoring_pipeline.py --seed 2026 --tag s2026

# AI
python train_gru.py --epochs 10 --seed 42 --tag _ai_s42
python train_gru.py --epochs 7  --seed 7  --tag _ai_s7
python train_gru.py --epochs 7  --seed 13 --tag _ai_s13
python train_gru.py --epochs 7  --seed 99   --hidden 192 --attn --tag _attn_s99
python train_gru.py --epochs 7  --seed 2024 --hidden 192 --attn --tag _attn_s2024
python train_transformer.py --epochs 10 --seed 42 --tag _tr

# Бледнинг
python blend.py --models s42+s7+s2026 gru_ai_s42+gru_ai_s7+gru_ai_s13 gru_attn_s99+gru_attn_s2024 tr \
                --output submission_final.csv
```
