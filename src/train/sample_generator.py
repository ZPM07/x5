import pandas as pd
import random
import copy
from collections import Counter
import ast
import re
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import unicodedata
from typing import List, Dict, Tuple, Optional

from utils.generation_config import brands_by_category, fake_brands


class Generator:
    def __init__(
        self,
        train_path: str = "data/train_brand.csv",
        val_path: str = "best_submit.csv",
        sep: str = ";",
        min_samples: int = 10,
        corruption_level: float = 0.83,
        model_name: str = 'distiluse-base-multilingual-cased-v2'
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.sep = sep
        self.min_samples = min_samples
        self.corruption_level = corruption_level

        # Инициализация модели
        self.model = SentenceTransformer(model_name)

        # Подготовка данных
        self.df = None
        self.df_val = None
        self.brand_counts = None
        self.rare_brands = None
        self.rows_with_brands = None
        self.category_embeddings = None

        # Инициализация корруптора
        self.corruptor = QueryCorruptor(corruption_level=self.corruption_level)

    def _load_and_preprocess(self):
        """Загружает и предобрабатывает данные."""
        self.df = pd.read_csv(self.train_path, sep=self.sep)
        self.df['annotation'] = self.df['annotation'].apply(ast.literal_eval)
        self.df['label'] = self.df['annotation'].apply(self._label_func)
        self.df["sample"] = self.df["sample"].apply(self._clean_text)

        self.df_val = pd.read_csv(self.val_path, sep=self.sep)
        self.df_val['annotation'] = self.df_val['annotation'].apply(ast.literal_eval)
        self.df_val['label'] = self.df_val['annotation'].apply(self._label_func)
        self.df_val["sample"] = self.df_val["sample"].apply(self._clean_text)

    @staticmethod
    def _label_func(row):
        res = []
        for elem in row:
            if elem[2] == "B-BRAND":
                res.append(1)
            elif elem[2] == "I-BRAND":
                res.append(2)
            else:
                res.append(0)
        return res

    @staticmethod
    def _remove_accents_latin_only(text):
        def repl(ch):
            if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
                decomp = unicodedata.normalize('NFD', ch)
                return ''.join(c for c in decomp if unicodedata.category(c) != 'Mn')
            return ch
        return ''.join(repl(ch) for ch in text)

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.replace("ë", "ё").replace("Ë", "Ё")
        text = Generator._remove_accents_latin_only(text)
        text = re.sub(r"\\n|\\t", "", text)
        text = ''.join(
            ch for ch in text
            if unicodedata.category(ch)[0] != 'S' or ch == '№'
        )
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
        text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ№!&0-9%\-–—,.'’_/ +]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_brands(self) -> List[str]:
        """Извлекает все бренды из разметки."""
        brand_words = []
        for sample, labels in zip(self.df["sample"], self.df["label"]):
            words = sample.split()
            i = 0
            while i < min(len(words), len(labels)):
                if labels[i] == 1:
                    start = i
                    i += 1
                    while i < len(labels) and labels[i] == 2:
                        i += 1
                    end = i - 1
                    brand_text = " ".join(words[start:end+1]).lower()
                    brand_words.append(brand_text)
                else:
                    i += 1
        return brand_words

    def _prepare_category_embeddings(self):
        """Создаёт эмбеддинги для категорий на основе брендов."""
        category_texts = {cat: " ".join(words) for cat, words in brands_by_category.items()}
        self.category_embeddings = {
            cat: self.model.encode(text) for cat, text in category_texts.items()
        }

    @staticmethod
    def _cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def _assign_category(self, text: str) -> str:
        emb = self.model.encode(text)
        sims = {
            cat: self._cosine_similarity(emb, cat_emb)
            for cat, cat_emb in self.category_embeddings.items()
        }
        return max(sims, key=sims.get)

    def _filter_rows_with_brands(self):
        """Отбирает строки, содержащие бренд и хотя бы одно слово без бренда."""
        self.rows_with_brands = self.df[
            self.df["label"].apply(lambda x: any(lbl == 1 for lbl in x) and any(lbl == 0 for lbl in x))
        ]

    def _identify_rare_brands(self):
        brand_words = self._extract_brands()
        self.brand_counts = Counter(brand_words)
        self.rare_brands = {
            b: c for b, c in self.brand_counts.items() if c < self.min_samples
        }

    def generate(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Генерирует обучающий и валидационный датасеты.
        Возвращает:
            train_dataset: список словарей {"text": ..., "labels": ...}
            val_dataset: список словарей {"text": ..., "labels": ...}
        """
        self._load_and_preprocess()
        self._prepare_category_embeddings()
        self._identify_rare_brands()
        self._filter_rows_with_brands()

        synthetic_samples = []

        # === 1. Аугментация редких брендов ===
        for brand, count in self.rare_brands.items():
            n_needed = self.min_samples - count
            category = self._assign_category(brand)
            if not category or category not in brands_by_category:
                continue

            category_brands = brands_by_category[category]

            for _ in range(n_needed):
                row = self.rows_with_brands.sample(1).iloc[0]
                words = row["sample"].split()
                labels = copy.deepcopy(row["label"])

                new_brand = random.choice(category_brands)
                new_brand = self.corruptor.corrupt_query(new_brand)
                new_brand_words = new_brand.split()
                new_labels = [1] + [2] * (len(new_brand_words) - 1)

                if random.random() < 0.5:
                    # Замена существующего бренда
                    brand_positions = self._find_brand_spans(labels)
                    if not brand_positions:
                        continue
                    start, end = random.choice(brand_positions)
                    words = words[:start] + new_brand_words + words[end+1:]
                    labels = labels[:start] + new_labels + labels[end+1:]
                else:
                    # Вставка
                    zero_positions = [i for i, lbl in enumerate(labels) if lbl == 0]
                    if not zero_positions:
                        continue
                    pos = random.choice(zero_positions)
                    insert_pos = pos if random.random() < 0.5 else pos + 1
                    words = words[:insert_pos] + new_brand_words + words[insert_pos:]
                    labels = labels[:insert_pos] + new_labels + labels[insert_pos:]

                labels = self._align_labels(words, labels)
                synthetic_samples.append({"text": " ".join(words), "labels": labels})

        # === 2. Негативные примеры с fake_brands ===
        for nb in fake_brands:
            emb = self.model.encode(nb)
            sims = {
                cat: self._cosine_similarity(emb, cat_emb)
                for cat, cat_emb in self.category_embeddings.items()
            }
            category = max(sims, key=sims.get)

            for _ in range(self.min_samples + 100):
                row = self.rows_with_brands.sample(1).iloc[0]
                words = row["sample"].split()
                labels = copy.deepcopy(row["label"])

                nb_words = nb.split()
                nb_labels = [0] * len(nb_words)

                if random.random() < 0.5:
                    # Вставка
                    zero_positions = [i for i, lbl in enumerate(labels) if lbl == 0]
                    if not zero_positions:
                        continue
                    pos = random.choice(zero_positions)
                    insert_pos = pos if random.random() < 0.5 else pos + 1
                    words = words[:insert_pos] + nb_words + words[insert_pos:]
                    labels = labels[:insert_pos] + nb_labels + labels[insert_pos:]
                else:
                    # Замена
                    zero_positions = [i for i, lbl in enumerate(labels) if lbl == 0]
                    if not zero_positions:
                        continue
                    pos = random.choice(zero_positions)
                    words = words[:pos] + nb_words + words[pos+1:]
                    labels = labels[:pos] + nb_labels + labels[pos+1:]

                labels = self._align_labels(words, labels)
                synthetic_samples.append({
                    "text": " ".join(words),
                    "labels": labels,
                    "is_negative": True,
                    "category": category
                })

        # === 3. Формируем итоговые датасеты ===
        train_original = [
            {"text": text, "labels": labels}
            for text, labels in zip(self.df['sample'], self.df['label'])
        ]

        val_dataset = [
            {"text": text, "labels": labels}
            for text, labels in zip(self.df_val['sample'], self.df_val['label'])
        ]

        synthetic_data = [{"text": s["text"], "labels": s["labels"]} for s in synthetic_samples]

        train_dataset = train_original + synthetic_data + val_dataset

        print(f"Train: {len(train_dataset)} примеров (включая синтетику)")
        print(f"Validation: {len(val_dataset)} примеров (только оригинал)")
        print(f"Синтетических примеров добавлено: {len(synthetic_samples)}")

        return train_dataset, val_dataset

    @staticmethod
    def _find_brand_spans(labels: List[int]) -> List[Tuple[int, int]]:
        spans = []
        i = 0
        while i < len(labels):
            if labels[i] == 1:
                start = i
                i += 1
                while i < len(labels) and labels[i] == 2:
                    i += 1
                end = i - 1
                spans.append((start, end))
            else:
                i += 1
        return spans

    @staticmethod
    def _align_labels(words: List[str], labels: List[int]) -> List[int]:
        if len(labels) < len(words):
            labels += [0] * (len(words) - len(labels))
        elif len(labels) > len(words):
            labels = labels[:len(words)]
        return labels


class QueryCorruptor:
    def __init__(self, corruption_level=0.03):
        self.corruption_level = corruption_level
        self.keyboard_typos = {
            'а': 'фис', 'б': 'ьи', 'в': 'цыа', 'г': 'пыр', 'д': 'ывщ',
            'е': 'кун', 'ё': 'кун', 'ж': 'эд', 'з': 'швэ', 'и': 'цшщ',
            'й': 'цыф', 'к': 'нуп', 'л': 'дро', 'м': 'ьт', 'н': 'кеп',
            'о': 'лпа', 'п': 'нрк', 'р': 'пзол', 'с': 'ьмыв', 'т': 'ьм',
            'у': 'кгщ', 'ф': 'йа', 'х': 'чзж', 'ц': 'ув', 'ч': 'сх',
            'ш': 'гщз', 'щ': 'шзд', 'ъ': 'э', 'ы': 'вад', 'ь': 'бю',
            'э': 'жъ', 'ю': 'ь.', 'я': 'ря'
        }
        self.phonetic_typos = {
            'а': 'о', 'о': 'а', 'е': 'и', 'и': 'е', 'й': 'и',
            'т': 'д', 'д': 'т', 'п': 'б', 'б': 'п', 'к': 'г', 'г': 'к',
            'з': 'с', 'с': 'з', 'в': 'ф', 'ф': 'в', 'ж': 'ш', 'ш': 'ж'
        }
        self.wrong_endings = {
            re.compile(r'ное$'): 'на',
            re.compile(r'ная$'): 'но',
            re.compile(r'ий$'): 'ей',
        }

    def _miss_character(self, word):
        if len(word) <= 3: return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos+1:]

    def _duplicate_character(self, word):
        if len(word) <= 2: return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos] + word[pos:]

    def _swap_adjacent(self, word):
        if len(word) <= 3: return word
        pos = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
        return ''.join(chars)

    def _keyboard_typo(self, word):
        if len(word) <= 1: return word
        pos = random.randint(0, len(word) - 1)
        char = word[pos]
        if char.lower() in self.keyboard_typos:
            replacement = random.choice(self.keyboard_typos[char.lower()])
            if char.isupper():
                replacement = replacement.upper()
            return word[:pos] + replacement + word[pos+1:]
        return word

    def _phonetic_typo(self, word):
        if len(word) <= 1: return word
        pos = random.randint(0, len(word) - 1)
        char = word[pos]
        if char.lower() in self.phonetic_typos:
            replacement = self.phonetic_typos[char.lower()]
            if char.isupper():
                replacement = replacement.upper()
            return word[:pos] + replacement + word[pos+1:]
        return word

    def _change_ending(self, word):
        for pattern, replacement in self.wrong_endings.items():
            if pattern.search(word):
                return pattern.sub(replacement, word)
        return word

    def _shorten_word(self, word):
        if len(word) >= 8 and random.random() < 0.3:
            cut = random.randint(1, min(3, len(word) - 4))
            return word[:-cut]
        return word

    def corrupt_word(self, word):
        distortions = [
            (self._miss_character, 2),
            (self._keyboard_typo, 3),
            (self._phonetic_typo, 2),
            (self._swap_adjacent, 1),
            (self._duplicate_character, 1),
            (self._shorten_word, 2),
            (self._change_ending, 1),
        ]
        choices, weights = zip(*distortions)
        chosen = random.choices(choices, weights=weights)[0]
        return chosen(word)

    def corrupt_query(self, query):
        if random.random() > self.corruption_level:
            return query
        words = query.split()
        corrupted = []
        for word in words:
            if random.random() < 0.4:
                corrupted.append(self.corrupt_word(word))
            else:
                corrupted.append(word)
        if len(corrupted) > 1 and random.random() < 0.1:
            j = random.randint(0, len(corrupted) - 2)
            corrupted[j] += corrupted.pop(j + 1)
        result = ' '.join(corrupted)
        if random.random() < 0.05:
            sym = random.choice('1234567890.,!')
            if random.random() < 0.5:
                result += sym
            else:
                p = random.randint(0, len(result))
                result = result[:p] + sym + result[p:]
        return result