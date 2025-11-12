# -*- coding: utf-8 -*-

# Text 1

"""3311_bajmuhamedov_arshin_pasechny_practice_BAD.ipynb

Original file is located at
    https://colab.research.google.com/drive/1H3Lq0DAWUHRgZv0vycPhZJZN29IP0uBx

Notebook для практики по дисциплине "Методы машинного обучения"
Выполнили студенты:
- Баймухамедов Рафаэль Русланович
- Аршин Александр Дмитриевич
- Пасечный Леонид Витальевич

Преподаватель
- Петруша Полина Георгиевна

Скачаем датасет с Яндекс.Диска
"""

# Code 2

import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = "https://disk.yandex.ru/d/V1sJpR-SUJ_b8A"

final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

download_response = requests.get(download_url)
with open('dataset.xlsx', 'wb') as f:
    f.write(download_response.content)

# Text 3

"""Прочитаем в датафрейм наш файл"""

# Code 4

from pathlib import Path
import pandas as pd

xlsx_path = "dataset.xlsx"
if not xlsx_path:
    raise FileNotFoundError("xlsx файл не найден")
print("Найден XLSX:", xlsx_path)

df = pd.read_excel(xlsx_path, sheet_name=0, header=[0,1])
print("Данные загружены в df")

# Text 5

"""Настроим pandas"""

# Code 6

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Text 7

"""Посмотрим на датафрейм"""

# Code 8

df.head()

import re
from collections import Counter, defaultdict

def clean(s):
    if s is None: return ""
    s = str(s).replace("\n"," ").replace("\xa0"," ").strip()
    return re.sub(r"\s+"," ", s)

flat = []
for top, sub in df.columns:
    top, sub = clean(top), clean(sub)
    name = sub if (not top or top.lower().startswith("unnamed")) else f"{top}__{sub}" if sub else top
    name = name.replace("ё","е")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[\\/:;,\"'()]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    name = name.lower()
    flat.append(name)

cnt = Counter(flat); seen = defaultdict(int); uniq = []
for n in flat:
    seen[n] += 1
    uniq.append(n if cnt[n] == 1 else f"{n}__{seen[n]}")

df.columns = uniq

to_rename = {
    "пищевые_вещества_макро-_и_микроэлементы": "пищевые_вещества_макро_и_микроэлементы",
    "минеральные_и_минерало-органические_природные_субстанции_цеолиты_гуминовые_кислоты":"минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты",
    "система_органов_костно-мышечная_сиситема": "система_органов_костно_мышечная_система",
    "система_органов_форма_выпуска":"форма_выпуска",
    "система_органов_продолжительность_приема":"продолжительность_приема",
    "система_органов_происхождение":"происхождение",
    "система_органов_сырье_растительное_животное_биологическое":"сырье"

}

df = df.rename(columns=to_rename)

print("Имена колонок (первые 50):")
for c in df.columns[:]:
    print("-", c)

df.head()