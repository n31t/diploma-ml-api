
  Моя рекомендация

  Отдать бэкенду exported-kazbert-detector-t060 — баланс recall (0.52/0.62) лучше, плюс среднее P(AI) точно на 0.5 даёт самые читаемые проценты на UI. Если бизнес-требование
  «не пропустить AI важнее, чем не обидеть человека» — тогда t055 (recall_ai=0.76 ценой recall_hum=0.38).

✻ Churned for 2m 43s

❯ где эти модели тут хранятся                                                                                                                                                

⏺ Bash(cd /Users/adilovamir/diploma/slop-detector
      echo "=== Полные пути ==="…)
  ⎿  === Полные пути ===
     /Users/adilovamir/diploma/slop-detector/models/exported-kazbert-detector-t055
     /Users/adilovamir/diploma/slop-detector/models/exported-kazbert-detector-t060
     … +45 lines (ctrl+o to expand)

⏺ Расположение моделей

  /Users/adilovamir/diploma/slop-detector/models/exported-kazbert-detector-t055
  /Users/adilovamir/diploma/slop-detector/models/exported-kazbert-detector-t060

  Каждая 423 MB, обе содержат всё что нужно:

  ┌───────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │                                   Файл                                    │                             Назначение                              │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ model.safetensors (422 MB)                                                │ Веса полной модели (LoRA merged in)                                 │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ config.json                                                               │ Архитектура + id2label: {0:"Human", 1:"AI"}                         │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ tokenizer.json, vocab.txt, tokenizer_config.json, special_tokens_map.json │ Токенайзер                                                          │
  ├───────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ calibration_meta.json                                                     │ Метаданные про baked threshold (для дебага, не нужен для инференса) │
  └───────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

  Как передать бэкенду

  Вариант 1 — отдать всю папку как есть:
  # zip для пересылки
  cd /Users/adilovamir/diploma/slop-detector/models
  zip -r exported-kazbert-detector-t060.zip exported-kazbert-detector-t060/
  Получится файл ~400 MB (safetensors уже хорошо упакован, ZIP мало что сжимает).

  Вариант 2 — залить на Hugging Face Hub (если бэкенд может тянуть оттуда):
  cd /Users/adilovamir/diploma/slop-detector/models/exported-kazbert-detector-t060
  huggingface-cli upload <твой-namespace>/kazbert-ai-detector-t060 .
  Тогда бэкенд просто AutoModelForSequenceClassification.from_pretrained("<namespace>/kazbert-ai-detector-t060") — без передачи файла.

  Вариант 3 — S3/любой объектный сторадж:
  Залить три файла что бэкенд реально использует: model.safetensors, config.json, tokenizer.json (+ vocab.txt если bert-tokenizer). Остальное необязательно. Это ~423 MB
  суммарно.

  Что бэкенду НЕ нужно

  - models/kazbert-repair-from-deepseek-pooler-fix/ — это был промежуточный PEFT-адаптер, уже merged в экспорт
  - models/kazbert-wikipedia-deepseek-kk-pooler-fix/ — исходник для repair, тоже больше не нужен
  - Базовая модель Eraly-ml/KazBERT — она тоже merged внутрь экспорта, отдельно ничего тянуть не надо

  Один путь, один from_pretrained, всё.

✻ Worked for 22s