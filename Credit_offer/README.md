# Задача 2 — Кредитный оффер

Метрика ROC-AUC. Итог: 0.766

# Запуск
```bash
python offer_response_pipeline.py
```

# Файлы
- eda_apps.py, eda_apps2.py — анализ
- experiment_offer.py — recency-веса и гиперпараметры
- experiment_adv.py — adversarial-валидация + importance weight
- experiment_drop.py — удаление времязависимых признаков
- experiment_blend_holdout.py — выбор веса бленда на честном time-holdout
