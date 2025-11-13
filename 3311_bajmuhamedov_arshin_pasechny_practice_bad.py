# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Notebook для практики по дисциплине "Методы машинного обучения"
# Выполнили студенты:
# - Баймухамедов Рафаэль Русланович
# - Аршин Александр Дмитриевич
# - Пасечный Леонид Витальевич
#
# Преподаватель
# - Петруша Полина Георгиевна

# %% [markdown]
# Скачаем датасет с Яндекс.Диска

# %%
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

# %% [markdown]
# Прочитаем в датафрейм наш файл

# %%
from pathlib import Path
import pandas as pd

xlsx_path = "dataset.xlsx"
if not xlsx_path:
    raise FileNotFoundError("xlsx файл не найден")
print("Найден XLSX:", xlsx_path)

df = pd.read_excel(xlsx_path, sheet_name=0, header=[0,1])
print("Данные загружены в df")

# %% [markdown]
# Настроим pandas

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# Посмотрим на датафрейм

# %%
df.head()

# %% [markdown]
# Соединим заголовки первого и второго уровня вместе

# %%
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

# %% [markdown]
# Переименуем некоторые столбцы

# %%
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

print("Имена колонок:")
for c in df.columns[:]:
    print("-", c)

df.head()

# %% [markdown]
# Сохранение изменений

# %%
# !pip -q install jupytext nbstripout

from google.colab import drive
drive.mount('/content/drive')

NOTEBOOK = "/content/drive/MyDrive/Colab Notebooks/3311_bajmuhamedov_arshin_pasechny_practice_BAD.ipynb"

cfg = '''formats = "ipynb,py:percent"
cell_metadata_filter = "-all"
notebook_metadata_filter = "kernelspec,jupytext"
'''
with open("/content/.jupytext.toml", "w", encoding="utf-8") as f:
    f.write(cfg)

import os, pathlib, time, textwrap, subprocess, json
ipynb_path = pathlib.Path(NOTEBOOK)
py_path = ipynb_path.with_suffix(".py")

if not ipynb_path.exists():
    raise FileNotFoundError(f"Не найден .ipynb: {ipynb_path}")

print("IPYNB:", ipynb_path)
print("PY:", py_path)

# !nbstripout "{NOTEBOOK}"

if py_path.exists():
    py_path.unlink()
# !jupytext --to py:percent "{NOTEBOOK}"

import datetime
stat = py_path.stat()
print("\nОбновлён .py:", py_path)
