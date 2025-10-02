# 🛒 X5 NER — Извлечение сущностей из поисковых запросов

**X5 NER** — это решение задачи *именованного распознавания сущностей (NER)* в поисковых запросах пользователей приложения **«Пятёрочка»**.  
Модель обучена извлекать сущности в формате BIO, включая:

- 🏷️ `TYPE`
- 🏷️ `BRAND`
- 🧪 `VOLUME`
- 🔢 `PERCENT`

---

## 🚀 Быстрый старт

### 📦 Установка

Проект использует [**uv**](https://github.com/astral-sh/uv) для управления зависимостями.

```bash
# Установка зависимостей из pyproject.toml
uv sync

# Установка проекта в режиме разработки
uv pip install -e .
```

## 🧪 Примеры использования
### 1. 🔬 Инференс в ноутбуке

Пример использования модели в Jupyter Notebook:
📁 notebooks/inference_example.ipynb

### 2. 🛠️ Генерация submission.csv через CLI

``` bash
python pipeline.py \
    --input data/raw/test.csv \
    --output data/processed/submission.csv
```

### 3. 🌐 Запуск API-сервиса
``` bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

После запуска сервис будет доступен по адресу:
🔗 http://localhost:8000

### Пример запроса:
``` bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": "кола 2л без сахара"}'
```

### Пример ответа:
``` json
[
  {"start_index": 0, "end_index": 4, "entity": "B-BRAND"},
  {"start_index": 5, "end_index": 7, "entity": "B-VOLUME"}
]
```
📌 При пустом входе ("") возвращается пустой список: [].

## Структура проекта

``` csharp
x5/
├── data/
│   ├── raw/                  # Исходные данные (например, test.csv)
│   └── processed/            # Результаты инференса (submission.csv)
├── models/                   # Предобученные модели (BERT + XGBoost)
├── notebooks/                # Jupyter-ноутбуки с примерами
├── src/                      # Основная логика проекта
│   ├── models/               # Загрузка/инференс моделей
│   └── utils/                # Утилиты и вспомогательные функции
├── api.py                    # FastAPI endpoint
├── pipeline.py               # CLI-скрипт для генерации submission
├── pyproject.toml            # Конфигурация проекта и зависимости
└── uv.lock                   # Лок-файл зависимостей (генерируется uv)
```

## 🤝 Авторы

Разработано командой **Мушкетёры 3.5 🏴‍☠️**