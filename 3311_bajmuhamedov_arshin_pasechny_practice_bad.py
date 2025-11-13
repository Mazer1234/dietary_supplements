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
# Соединим заголовки первого и второго уровня вместе. Также уберем пробелы между словами в столбцах, заменив их на "_" и приведем названия столбцов к нижнему регистру.

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
    "система_органов_сырье_растительное_животное_биологическое":"сырье",
    "система_органов_сердечно-сосудистая_система":"система_органов_сердечно_сосудистая_система"
}

df = df.rename(columns=to_rename)

print("Имена колонок:")
for c in df.columns[:]:
    print("-", c)

df.head()

# %% [markdown]
# Создаем 2 новых столбца:
# - Рекомендации по применению
# - Продолжительность приема
#
# Информацию для них берем из столбца Этикетка, затем отчищаем оттуда взятую инфу.

# %%
df['рекомендации_по_применению'] = 'a'
df['продолжительность_приема'] = 'b'

for row in range(len(df)):
  string = str(df.at[row, 'этикетка'])

  dot = string.find(".")+1
  str_for_et = str(df.at[row, 'этикетка'])[string.find(".", dot)+1::]

  value_1 = string[string.find("рекомендации_по_применению:"):dot]
  df.at[row, 'рекомендации_по_применению'] = value_1

  value_2 = string[string.find("продолжительность_приема"):string.find(".", dot)]
  df.at[row, 'продолжительность_приема'] = value_2

  df.at[row, 'этикетка'] = str_for_et

# %% [markdown]
# Срок годности преобразуем из лет в месяцы

# %% [markdown]
# Посмотрим уникальные значения до изменений

# %%
print(df['срок_годности'].unique())

# %%
bad_val = {"1 год, 2 года" : 24,
           "1 год, 2 месяца": 14,
           "1,5 года": 18,
           "15 суток": 0.5,
           "2 года": 24,
           "2 года, 1 год": 24,
           "2 года, 1,5 года": 24,
           "2,5 года": 30,
           "3 года": 36,
           "3,5 года": 42,
           "4 года": 48,
           "5 лет": 60,
           "1 год": 12
           }
for row in range(len(df)):
  val = str(df.at[row, 'срок_годности'])
  if val in bad_val.keys():
    df.at[row, 'срок_годности'] = bad_val[val]
  else:
    if ('меся' in str(df.at[row, 'срок_годности'])):
      string = str(df.at[row, 'срок_годности'])
      str_res = string[:string.index(" ")]
      if (str_res != ''):
        if (',' in str(df.at[row, 'срок_годности'])):
          str_res = str_res.replace(',', '.')
        df.at[row, 'срок_годности'] = float(str_res)

# %% [markdown]
# Посмотрим уникальные значения после (Все преобразовалось)

# %%
print(df['срок_годности'].unique())

# %% [markdown]
# Также меняем столбец возраст детей. Меняем года на месяцы
# Будет указано одно значение (12 например). Оно будет означать, что для детей от 12 месяцев.

# %%
print(df.columns)

# %%
bad_val = {"от 11 лет" : 132,
           "от 12 лет": 144,
           "от 14 лет": 168,
           "от 3 месяцев": 3,
           "от 7 лет": 84,
           "с рождения": 0,
           "от 1,5 лет": 24,
           "от 3 лет": 36,
           "от 4 лет": 48,
           "от 5 лет": 60,
           "от 1 года": 12
           }
for i in range(len(df)):
  val = str(df.at[i, 'группа_населения_возраст_детей'])
  if val in bad_val.keys():
    df.at[i, 'группа_населения_возраст_детей'] = bad_val[val]
print(df['группа_населения_возраст_детей'].unique())
# %% [markdown]
# Исправим орфографические ошибки в строках

# %% [markdown]
# Посмотрим на уникальные значения в некоторых столбцах

# %%
print(df["пищевые_вещества_белки_пептиды_аминокислоты_нуклеиновые_кислоты"].unique())
print(df["минорные_компоненты_растений_алкалоиды"].unique())
print(df["пищевые_вещества_углеводы_и_продукты_их_переработки"].unique())
print(df["минорные_компоненты_растений_гидроксикоричные_кислоты"].unique())
print(df["минорные_компоненты_растений_ферменты"].unique())
print(df["минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты"].unique())
print(df["система_органов_для_беременных_кормящих_и_планирующих_беременность"].unique()),
print(df["система_органов_костно_мышечная_система"].unique()),
print(df["система_органов_нервная_система"].unique()),
print(df["система_органов_иммунная_система"].unique()),
print(df["система_органов_дерматологические_бад"].unique()),
print(df["система_органов_сердечно_сосудистая_система"].unique()),
print(df["система_органов_противопаразитарные_бад"].unique()),
print(df["система_органов_дыхательная_система"].unique()),
print(df["система_органов_противомикробные_бад"].unique()),


# %% [markdown]
# Напишем функцию, которая заменяет значение из списка в строке столбца на заданное значение

# %%
def replace_exact(df, col, variants, target):
    df.loc[df[col].isin(variants), col] = target

pairs = [
    ["пищевые_вещества_белки_пептиды_аминокислоты_нуклеиновые_кислоты", ["аминоксилоты"], "аминокислоты"],
    ["минорные_компоненты_растений_алкалоиды", ["алкалод", "алкалоид"], "алкалоиды"],
    ["пищевые_вещества_углеводы_и_продукты_их_переработки", ["полисахарид", "полисахарилы", "полисхариды"], "полисахариды"],
    ["минорные_компоненты_растений_гидроксикоричные_кислоты", ["гидрокор"], "гидроксикор"],
    ["минорные_компоненты_растений_ферменты", ["фермент"], "ферменты"],
    ["минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты", ["цеолит"], "цеолиты"],
    ["система_органов_для_беременных_кормящих_и_планирующих_беременность", ["берем"], "беременные"],
    ["система_органов_костно_мышечная_система", ["суст", "суств"], "суставы"],
    ["система_органов_нервная_система", ["невр", "неврная", "нерврная", "нерв"], "нервная"],
    ["система_органов_иммунная_система", ["имм", "имммун", "иммун"], "иммунитет"],
    ["система_органов_дерматологические_бад", ["коэа"], "кожа"],
    ["система_органов_сердечно_сосудистая_система", ["серд"], "сердце"],
    ["система_органов_противопаразитарные_бад", ["паразит"], "паразиты"],
    ["система_органов_дыхательная_система", ["легк"], "легкие"],
    ["система_органов_противомикробные_бад", ["бакт", "бактер"], "бактерия"],
    ["система_органов_противомикробные_бад", ["вир"], "вирус"],
    ["система_органов_противомикробные_бад", ["вир"], "вирус"],
    ["система_органов_противомикробные_бад", ["грию"], "гриб"]
]

for i in range(len(pairs)):
    replace_exact(df, pairs[i][0],pairs[i][1], pairs[i][2])

# %% [markdown]
# Проверим теперь уникальные значения этих же столбцов

# %%
print(df["пищевые_вещества_белки_пептиды_аминокислоты_нуклеиновые_кислоты"].unique())
print(df["минорные_компоненты_растений_алкалоиды"].unique())
print(df["пищевые_вещества_углеводы_и_продукты_их_переработки"].unique())
print(df["минорные_компоненты_растений_гидроксикоричные_кислоты"].unique())
print(df["минорные_компоненты_растений_ферменты"].unique())
print(df["минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты"].unique())
print(df["система_органов_для_беременных_кормящих_и_планирующих_беременность"].unique()),
print(df["система_органов_костно_мышечная_система"].unique()),
print(df["система_органов_нервная_система"].unique()),
print(df["система_органов_иммунная_система"].unique()),
print(df["система_органов_дерматологические_бад"].unique()),
print(df["система_органов_сердечно_сосудистая_система"].unique()),
print(df["система_органов_противопаразитарные_бад"].unique()),
print(df["система_органов_дыхательная_система"].unique()),
print(df["система_органов_противомикробные_бад"].unique())

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
