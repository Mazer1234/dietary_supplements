# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

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
#

# %%
from pathlib import Path
import pandas as pd

xlsx_path = "dataset.xlsx"
if not xlsx_path:
    raise FileNotFoundError("xlsx файл не найден")
print("Найден XLSX:", xlsx_path)

df = pd.read_excel(xlsx_path, sheet_name=0, header=[0,1])
print("Данные загружены в df")


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


# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# %%
print(df['этикетка'])

# %% [markdown]
# Создаем 2 новых столбца:
# - Рекомендации по применению
# - Продолжительность приема
#
# Информацию для них берем из столбца Этикетка, затем отчищаем оттуда взятую инфу.

# %%
print(str(df.at[0, 'этикетка'])[:str(df.at[0, 'этикетка']).find(".")])

# %%
df['рекомендации_по_применению'] = 'a'
df['продолжительность_приема'] = 'b'
dot = 0


for row in range(len(df)):
  string = str(df.at[row, 'этикетка'])

  dot = string.find(".")+1
  str_for_et = str(df.at[row, 'этикетка'])[string.find(".", dot)+1::]
  value_1 = ""

  if "Рекомендации по применению" not in string:
    df.at[row, 'рекомендации_по_применению'] = None
  elif value_1 == "":
    index = 0
    current_string = string[string.find("Рекомендации по применению")::]
    while True:
      if current_string[index+1] != ".":
        index+=1
      else:
        break
    value_1 = string[string.find("Рекомендации по применению"):string.find("Рекомендации по применению")+index+1]
  df.at[row, 'рекомендации_по_применению'] = value_1

  if (string[string.find("Продолжительность приема")+1] == '.'):
    value_2 = string[string.find("Продолжительность приема"):string.find(".", dot+1)]
  else:
    value_2 = string[string.find("Продолжительность приема"):string.find(".", dot)]
  if "Продолжительность приема" not in string:
    df.at[row, 'продолжительность_приема'] = None
  elif value_2 == "":
      index = 0
      current_string = string[string.find("Продолжительность приема")::]
      while True:
        if current_string[index+1] != ".":
          index+=1
        else:
          break
      value_2 = string[string.find("Продолжительность приема"):string.find("Продолжительность приема")+index+1]
      df.at[row, 'продолжительность_приема'] = value_2
  else:
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
# Выделим дополнительный столбец кол-во раз в день. У нас получилось, что для некоторых проще поставить nan, чем сильно усложнять логику, таких всего 19.

# %%
from numpy import nan

df['количество_раз_в_день'] = 'a'
df['количество_таблеток_за_прием'] = 'b'
df['продолжительность'] = 'c'
num = '0123456789'
for i in range(len(df)):
  string = str(df.at[i, 'рекомендации_по_применению'])
  if 'раз' not in string and '-' in string and string[string.find("-") + 1] in num:
    df.at[i, 'количество_раз_в_день'] = int(string[string.find("-") + 1])
  elif 'раз' not in string:
    df.at[i, 'количество_раз_в_день'] = 1
  else:
    val_amount = string[string.find(' раз')-1:string.find(' раз')]
    if (val_amount in num):
      df.at[i, 'количество_раз_в_день'] = int(val_amount[::-1])
    else:
      df.at[i, 'количество_раз_в_день'] = nan


# %%
print(df['количество_раз_в_день'].isna().sum())

# %% [markdown]
# Выделим продолжительность(продолжительность приема, но только одним числом без лишних слов). Получилось 600 записей без Продолжительности приема (на этикетке не было)

# %%
num = '0123456789'

for i in range(len(df)):
  string = str(df.at[i, 'продолжительность_приема'])
  if 'мес' in string and string[string.find(' мес') - 1:string.find(' мес')] in num:
    df.at[i, 'продолжительность'] = int(string[string.find(' мес') - 1:string.find(' мес')])
  elif 'недел' in string and string[string.find(' недел') - 1:string.find(' недел')] in num:
    df.at[i, 'продолжительность'] = int(string[string.find(' недел') - 1:string.find(' недел')])
  elif 'дней' in string:
    if '-' in string[string.find(' дней') - 2:string.find(' дней')]:
      df.at[i, 'продолжительность'] = int(string[string.find(' дней') - 1:string.find(' дней')])
    else:
      df.at[i, 'продолжительность'] = int(string[string.find(' дней') - 2:string.find(' дней')])
  elif 'дня' in string:
    if '-' in string[string.find(' дня') - 2:string.find(' дня')] or ' ' in string[string.find(' дня') - 2:string.find(' дня')]:
      df.at[i, 'продолжительность'] = int(string[string.find(' дня') - 1:string.find(' дня')])
    else:
      df.at[i, 'продолжительность'] = int(string[string.find(' дня') - 2:string.find(' дня')])
  else:
    df.at[i, 'продолжительность'] = nan

# %%
print(df['продолжительность'].isna().sum())

# %% [markdown]
# Выделим кол-во таблеток за прием

# %%

# %%
for i in range(len(df)):
  string = str(df.at[i, 'рекомендации_по_применению'])
  string = string[string.find(":")::]
  if ' по ' in string and string[string.find("по")+3] in num:
     df.at[i, 'количество_таблеток_за_прием'] = int(string[string.find("по")+3])
  else:
     df.at[i, 'количество_таблеток_за_прием'] = nan

# %%
print(df['количество_таблеток_за_прием'].unique())

# %%
df["общее_количество_лекарств_на_курс"] = 'a'

for row in range(len(df)):
  if pd.notna(df.at[row, 'количество_таблеток_за_прием']) and pd.notna(df.at[row, 'продолжительность']) and pd.notna(df.at[row, 'количество_раз_в_день']):
    df.at[row, "общее_количество_лекарств_на_курс"] = int(df.at[row, 'количество_таблеток_за_прием']) * int(df.at[row, 'продолжительность']) * int(df.at[row, 'количество_раз_в_день'])
  else:
      df.at[row, "общее_количество_лекарств_на_курс"] = nan

# %%
print(df.isna().sum())

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


# %%
