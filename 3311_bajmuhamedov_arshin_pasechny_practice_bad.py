# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all,tags
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
import numpy as np

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
pd.set_option('display.max_rows', None)

# %% [markdown]
# # Предобработка данных

# %% [markdown]
# Посмотрим на датафрейм до какой-либо предобработки данных

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
    "система_органов_сырье_растительное_животное_биологическое":"сырье",
    "система_органов_сердечно-сосудистая_система":"система_органов_сердечно_сосудистая_система",
    "система_органов_происхождение_природное_синтетическое":"происхождение_природное_синтетическое"
}

df = df.rename(columns=to_rename)

# %% [markdown]
# Создадим новый столбец "рекомендации_по_применению". Информацию для них берем из столбца Этикетка, затем отчищаем оттуда взятую инфу.

# %%
df['рекомендации_по_применению'] = pd.NA
dot = 0

for row in range(len(df)):
    string = str(df.at[row, 'этикетка'])
    value_1 = ""

    dot = string.find(".")+1
    second_dot_index = string.find(".", dot)

    if second_dot_index != -1:
        str_for_et = string[second_dot_index+1::].strip()
    else:
        str_for_et = string

    if "Рекомендации по применению" not in string:
        df.at[row, 'рекомендации_по_применению'] = None

    else:
        start_index = string.find("Рекомендации по применению")
        current_string = string[start_index:]
        duration_index = current_string.find("Продолжительность")

        if duration_index != -1:
            end_index = start_index + duration_index
            value_1 = string[start_index:end_index].strip()

            if value_1.endswith((':',' ','.')):
                value_1 = value_1[:-1].strip()

        else:
            index = 0
            while True:
                if index + 1 >= len(current_string):
                    value_1 = current_string.strip()
                    break

                if current_string[index+1] != ".":
                    index += 1
                else:
                    value_1 = string[start_index : start_index + index + 2].strip()
                    break

    df.at[row, 'рекомендации_по_применению'] = value_1
    df.at[row, 'этикетка'] = str_for_et

# %% [markdown]
# Создадим новый столбцы:
# - Количество единиц на прием
# - Количество приемов в день
# - Суммарное количество единиц за период (Количество единиц на приеме * Количество приемов в день * Продолжительность приема)

# %%
df["количество_единиц_на_прием"] = pd.NA
df["количество_приемов_в_день"] = pd.NA
df["суммарное_количество_единиц_за_период"] = pd.NA

re_po_range = re.compile(r'по\s*(\d+)\s*-\s*(\d+)', flags=re.IGNORECASE)
re_po_single = re.compile(r'по\s*(\d+)(?!\s*-\s*\d+)', flags=re.IGNORECASE)
re_grams = re.compile(r'\((\d+)\s*г\)', flags=re.IGNORECASE)
re_times_range = re.compile(r'(\d+)\s*-\s*(\d+)\s*раз', flags=re.IGNORECASE)
re_times_single = re.compile(r'(\d+)\s*раз', flags=re.IGNORECASE)
re_grams_inline = re.compile(r'\b(\d+)\s*(?:г\b|грамм\w*)', flags=re.IGNORECASE)
re_ml = re.compile(r'\b(\d+)\s*мл\b(?!\s*питьевой воды)', flags=re.IGNORECASE)


for i in range(len(df)):
    string_raw = str(df.at[i, 'рекомендации_по_применению'])
    string = string_raw.lower()
    units = np.nan
    times = np.nan

    # количество_единиц_на_прием
    m = re_po_range.search(string)
    if m:
        units = int(m.group(2))
    else:
        m = re_po_single.search(string)
        if m:
            units = int(m.group(1))
        else:
            if 'пакет' in string:
                idx = string.find('пакет')
                left_window = string[max(0, idx-12):idx]
                m = re.search(r'(\d+)\s*-\s*(\d+)', left_window)
                if m:
                    units = int(m.group(2))
                else:
                    m = re.search(r'(\d+)', left_window)
                    if m:
                        units = int(m.group(1))
            if np.isnan(units):
                m = re_grams.search(string)
                if m:
                    units = int(m.group(1))
            if np.isnan(units):
                m = re_grams_inline.search(string)
                if m:
                    units = int(m.group(1))
            if np.isnan(units):
                m = re_ml.search(string)
                if m:
                    units = int(m.group(1))
    df.at[i, 'количество_единиц_на_прием'] = units

    # количество_приемов_в_день

    times = 1
    m = re_times_range.search(string)
    if m:
        times = int(m.group(2))
    else:
        m = re_times_single.search(string)
        if m:
            times = int(m.group(1))
    df.at[i, 'количество_приемов_в_день'] = times

# Заменим некоторые неправильно обработанные строки вручную
df.loc[39, 'количество_единиц_на_прием'] = 1
df.loc[59, 'количество_единиц_на_прием'] = 2
df.loc[91, 'количество_единиц_на_прием'] = 1
df.loc[102, 'количество_единиц_на_прием'] = 6
df.loc[178, 'количество_единиц_на_прием'] = 2
df.loc[215, 'количество_единиц_на_прием'] = 2
df.loc[222, 'количество_единиц_на_прием'] = 5
df.loc[723, 'количество_единиц_на_прием'] = 4
df.loc[928, 'количество_единиц_на_прием'] = 1
df.loc[1207, 'количество_единиц_на_прием'] = 3
df.loc[1219, 'количество_единиц_на_прием'] = 1
df.loc[1262, 'количество_единиц_на_прием'] = 1
df.loc[1263, 'количество_единиц_на_прием'] = 2
df.loc[1264, 'количество_единиц_на_прием'] = 3
df.loc[1265, 'количество_единиц_на_прием'] = 3
df.loc[1289, 'количество_единиц_на_прием'] = 1
df.loc[1492, 'количество_единиц_на_прием'] = 1
df.loc[1693, 'количество_единиц_на_прием'] = 1
df.loc[1785, 'количество_единиц_на_прием'] = 1
df.loc[1786, 'количество_единиц_на_прием'] = 2
df.loc[1855, 'количество_единиц_на_прием'] = 6
df.loc[1884, 'количество_единиц_на_прием'] = 10
df.loc[1991, 'количество_единиц_на_прием'] = 2
df.loc[2117, 'количество_единиц_на_прием'] = 1
df.loc[2325, 'количество_единиц_на_прием'] = 3
df.loc[2391, 'количество_единиц_на_прием'] = 1
df.loc[2487, 'количество_единиц_на_прием'] = 6
df.loc[2517, 'количество_единиц_на_прием'] = 2
df.loc[2518, 'количество_единиц_на_прием'] = 4
df.loc[2523, 'количество_единиц_на_прием'] = 4
df.loc[2524, 'количество_единиц_на_прием'] = 2
df.loc[2531, 'количество_единиц_на_прием'] = 5
df.loc[2545, 'количество_единиц_на_прием'] = 8
df.loc[2598, 'количество_единиц_на_прием'] = 1
df.loc[2675, 'количество_единиц_на_прием'] = 3
df.loc[2724, 'количество_единиц_на_прием'] = 1
df.loc[2729, 'количество_единиц_на_прием'] = 2
df.loc[2843, 'количество_единиц_на_прием'] = 1
df.loc[2857, 'количество_единиц_на_прием'] = 3
df.loc[2859, 'количество_единиц_на_прием'] = 3
df.loc[2889, 'количество_единиц_на_прием'] = 1
df.loc[2894, 'количество_единиц_на_прием'] = 1
df.loc[2895, 'количество_единиц_на_прием'] = 1
df.loc[2900, 'количество_единиц_на_прием'] = 2
df.loc[2902, 'количество_единиц_на_прием'] = 3
df.loc[2937, 'количество_единиц_на_прием'] = 2
df.loc[2951, 'количество_единиц_на_прием'] = 1
df.loc[3028, 'количество_единиц_на_прием'] = 3
df.loc[3314, 'количество_единиц_на_прием'] = 5
df.loc[3470, 'количество_единиц_на_прием'] = 2
df.loc[3649, 'количество_единиц_на_прием'] = 1
df.loc[3660, 'количество_единиц_на_прием'] = 1
df.loc[3960, 'количество_единиц_на_прием'] = 1
df.loc[3963, 'количество_единиц_на_прием'] = 8
df.loc[3965, 'количество_единиц_на_прием'] = 3
df.loc[4026, 'количество_единиц_на_прием'] = 1
df.loc[4103, 'количество_единиц_на_прием'] = 1
df.loc[4141, 'количество_единиц_на_прием'] = 2

df.loc[22, 'количество_приемов_в_день'] = 4
df.loc[39, 'количество_приемов_в_день'] = 6
df.loc[91, 'количество_приемов_в_день'] = 2
df.loc[102, 'количество_приемов_в_день'] = 6
df.loc[319, 'количество_приемов_в_день'] = 3


# %% [markdown]
# Напишем функцию, выводящую суммарную информацию о датафрейме

# %%
def print_info(df):
    analysis = []

    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        missing_pct = (df[col].isna().sum() / len(df) * 100).round(2)

        unique_values = df[col].dropna().unique()
        if len(unique_values) <= 5:
            unique_display = list(unique_values)
        else:
            unique_display = ">5 unique values"

        analysis.append({
            'столбец': col,
            'тип': str(dtype),
            'уникальных': unique_count,
            'пропущено %': f"{missing_pct}%",
            'уникальные значения': unique_display
        })

    result_df = pd.DataFrame(analysis)

    from tabulate import tabulate
    print(tabulate(result_df, headers='keys', tablefmt='grid', showindex=False))


# %% [markdown]
# Напишем функцию, которая заменяет значение из списка в строке столбца на заданное значение. Таким образом, заменим:
# - орфографические ошибки
# - продолжительность приёма в значение месяца по максимальному значению
# - срок годности в месяцы
# - возраст детей в месяцы
# - столбцы с двумя уникальными значениями в бинарные

# %%
def replace_exact(df, col, variants, target):
    df.loc[df[col].isin(variants), col] = target

pairs = [
    # Исправление орфографических ошибок
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
    ["система_органов_противомикробные_бад", ["грию"], "гриб"],

    # Переведем продолжительность приема в месяцы
    ["продолжительность_приема", ["1 месяц и менее", "постоянно"], "1"],
    ["продолжительность_приема", ["1-2 месяца"], "2"],
    ["продолжительность_приема", ["3 месяца"], "3"],
    ["продолжительность_приема", ["6 месяцев"], "6"],
    ["продолжительность_приема", ["9 месяцев"], "9"],
    ["продолжительность_приема", ["не указано"], pd.NA],

    # Переведем срок годности в месяцы
    ["срок_годности", ["1 год"], "12"],
    ["срок_годности", ["1 год, 2 месяца"], "14"],
    ["срок_годности", ["1,5 года"], "18"],
    ["срок_годности", ["15 суток"], "0.5"],
    ["срок_годности", ["1 год, 2 года", "2 года", "2 года, 1 год", "2 года, 1,5 года"], "24"],
    ["срок_годности", ["2,5 года"], "30"],
    ["срок_годности", ["3 года"], "36"],
    ["срок_годности", ["3,5 года"], "42"],
    ["срок_годности", ["4 года"], "48"],
    ["срок_годности", ["5 лет"], "60"],
    ["срок_годности", ["1 месяц"], "1"],
    ["срок_годности", ["2 месяца"], "2"],
    ["срок_годности", ["2,5 месяца"], "2.5"],
    ["срок_годности", ["3 месяца"], "3"],
    ["срок_годности", ["6 месяцев"], "6"],
    ["срок_годности", ["8 месяцев"], "8"],
    ["срок_годности", ["11 месяцев"], "11"],
    ["срок_годности", ["14 месяцев"], "14"],
    ["срок_годности", ["19 месяцев"], "19"],
    ["срок_годности", ["20 месяцев"], "20"],
    ["срок_годности", ["21 месяц"], "21"],
    ["срок_годности", ["25 месяцев"], "25"],
    ["срок_годности", ["28 месяцев"], "28"],
    ["срок_годности", ["32 месяца"], "32"],

    # Переведем возраст детей в месяцы
    ["группа_населения_возраст_детей", ["с рождения"], "0"],
    ["группа_населения_возраст_детей", ["от 3 месяцев"], "3"],
    ["группа_населения_возраст_детей", ["от 1 года"], "12"],
    ["группа_населения_возраст_детей", ["от 1,5 лет"], "24"],
    ["группа_населения_возраст_детей", ["от 3 лет"], "36"],
    ["группа_населения_возраст_детей", ["от 4 лет"], "48"],
    ["группа_населения_возраст_детей", ["от 5 лет"], "60"],
    ["группа_населения_возраст_детей", ["от 7 лет"], "84"],
    ["группа_населения_возраст_детей", ["от 11 лет"], "132"],
    ["группа_населения_возраст_детей", ["от 12 лет"], "144"],
    ["группа_населения_возраст_детей", ["от 14 лет"], "168"],

    # Преобразуем форму выпуска БАДа в новые значения
    ["форма_выпуска", ["таблетки","капсулы","пастилки","пилюли","драже","леденцы","плитки","таблетки, капсулы", "капсулы, пилюли","таблетки, капсулы, пастилки","таблетки, пилюли","капсулы, пастилки"], "твердое"],
    ["форма_выпуска", ["порошок","гранулы","порошок, гранулы"], "сыпучее"],
    ["форма_выпуска", ["гели","пасты","желе"], "полутвердое"],
    ["форма_выпуска", ["растворы","суспензия"], "жидкое"],
    ["форма_выпуска", ["сбор"], "сборы"],
    ["форма_выпуска", ["таблетки, порошок","таблетки, капсулы, порошок","капсулы, порошок","капсулы, порошок ", "таблетки, порошок, капсулы, гранулы"], "твердое, сыпучее"],
    ["форма_выпуска", ["капсулы, растворы","таблетки, капсулы, растворы","таблетки, растворы"], "твердое, жидкое"],
    ["форма_выпуска", ["таблетки, капсулы, сбор","капсулы, сбор"], "твердое, сборы"],
    ["форма_выпуска", ["капсулы, порошок, пасты"], "твердое, сыпучее, полутвердое"],
    ["форма_выпуска", ["раствор, сбор"], "жидкое, сборы"],
]

for i in range(len(pairs)):
    replace_exact(df, pairs[i][0],pairs[i][1], pairs[i][2])

# %% [markdown]
# Объединение группы столбцов:
#
# - J-X: Биологически_активные_вещества
# - Y-AL: Системы_органов
# - AQ-AU: Группа_населения
#
# Они заносятся в отдельный датафрей и в дальнейшем добавляются к основному датафрейму

# %% [markdown]
# Сделаем список столбцов которые мы хотим объединить
# 1. биологически_вещества
# 2. системы_органов
# 3. группа_населения

# %%
df_copy=df.copy()
biolog_columns = [
    "пищевые_вещества_витамины_витаминоподобные_вещества_и_коферменты",
    "пищевые_вещества_макро_и_микроэлементы",
    "пищевые_вещества_жиры_жироподобные_вещества_и_их_производные",
    "пищевые_вещества_белки_пептиды_аминокислоты_нуклеиновые_кислоты",
    "пищевые_вещества_углеводы_и_продукты_их_переработки",
    "минорные_компоненты_растений_фенольные_соединения",
    "минорные_компоненты_растений_алкалоиды",
    "пробиотики_в_монокультурах_и_ассоциациях_пробиотические_микроорганизмы",
    "минорные_компоненты_растений_сапонины",
    "минорные_компоненты_растений_терпеноиды",
    "минорные_компоненты_растений_естественные_метаболиты_и_стимуляторы_метаболизма",
    "минорные_компоненты_растений_гидроксикоричные_кислоты",
    "минорные_компоненты_растений_ферменты",
    "минорные_компоненты_растений_дубильные_вещества",
    "минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты"
]

sys_org = []
for col in df.columns:
    if col.startswith("система_органов"):
        sys_org.append(col)

group_people = []
for col in df.columns:
   if col.startswith("группа_населения"):
    group_people.append(col)


# %% [markdown]
# Функция для объединения столбцов

# %%
def combine_columns(df, cols, name_map=None):
    def get_values(row):
        out = []
        for col in cols:
            val = row[col]
            if isinstance(val, (int, float)) and val == 1:
                pretty = name_map[col] if name_map and col in name_map else col
                out.append(pretty)
            elif isinstance(val, str) and val.strip():
                out.append(val.strip())
        if not out:
            return None

        return ", ".join(dict.fromkeys(out))
    return df.apply(get_values, axis=1)


# %%
df_bi=combine_columns(df_copy, biolog_columns, "биологически_активные_вещества")
df_sy=combine_columns(df_copy, sys_org, "системы_органов")
df_gr=combine_columns(df_copy, group_people, "группа_населения")

df_save=pd.DataFrame({
    "биологически_активные_вещества": df_bi,"системы_органы": df_sy,"группы_населения": df_gr})

# %% [markdown]
# Теперь, после создания объединенного столбца, для понимания корреляции, преобразуем значения столбцов, которые позже удалим, в бинарные

# %%
pairs_to_bin = [
    ["происхождение", ["иностранное"], "0"],
    ["происхождение", ["отечественное"], "1"],
    ["пищевые_вещества_витамины_витаминоподобные_вещества_и_коферменты", ["вит"], "1"],
    ["пищевые_вещества_макро_и_микроэлементы", ["элементы"], "1"],
    ["пищевые_вещества_белки_пептиды_аминокислоты_нуклеиновые_кислоты", ["аминокислоты"], "1"],
    ["минорные_компоненты_растений_фенольные_соединения", ["фенольн"], "1"],
    ["минорные_компоненты_растений_алкалоиды", ["алкалоиды"], "1"],
    ["пробиотики_в_монокультурах_и_ассоциациях_пробиотические_микроорганизмы", ["пробиотики"], "1"],
    ["пищевые_вещества_углеводы_и_продукты_их_переработки", ["полисахариды"], "1"],
    ["минорные_компоненты_растений_сапонины", ["сапонины"], "1"],
    ["минорные_компоненты_растений_терпеноиды", ["терпен"], "1"],
    ["минорные_компоненты_растений_естественные_метаболиты_и_стимуляторы_метаболизма", ["ест"], "1"],
    ["минорные_компоненты_растений_гидроксикоричные_кислоты", ["гидроксикор"], "1"],
    ["минорные_компоненты_растений_ферменты", ["ферменты"], "1"],
    ["минорные_компоненты_растений_дубильные_вещества", ["дуб"], "1"],
    ["минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты", ["цеолиты"], "1"],
    ["система_органов_для_беременных_кормящих_и_планирующих_беременность", ["беременные"], "1"],
    ["система_органов_костно_мышечная_система", ["суставы"], "1"],
    ["система_органов_нервная_система", ["нервная"], "1"],
    ["система_органов_иммунная_система", ["иммунитет"], "1"],
    ["система_органов_пищеварительный_тракт_и_обмен_веществ", ["жкт"], "1"],
    ["система_органов_мочеполовая_система", ["почки"], "1"],
    ["система_органов_дерматологические_бад", ["кожа"], "1"],
    ["система_органов_органы_чувств", ["глаза"], "1"],
    ["система_органов_сердечно_сосудистая_система", ["сердце"], "1"],
    ["система_органов_противоопухолевые_бад", ["онко"], "1"],
    ["система_органов_противопаразитарные_бад", ["паразиты"], "1"],
    ["система_органов_кровь_и_система_кроветворения", ["кровь"], "1"],
    ["группа_населения_предназначен_для_детей", ["дети"], "1"],
    ["группа_населения_предназначен_для_взрослых", ["взрослые"], "1"],
    ["группа_населения_пожилые", ["пожилые"], "1"]
]

for i in range(len(pairs_to_bin)):
    replace_exact(df, pairs_to_bin[i][0],pairs_to_bin[i][1], pairs_to_bin[i][2])

# %% [markdown]
# Изменим тип некоторых столбцов

# %%
df["продолжительность_приема"] = (pd.to_numeric(df["продолжительность_приема"], errors="coerce").astype("Int64"))
df["срок_годности"] = (pd.to_numeric(df["срок_годности"],errors="coerce").astype("Float64"))
df["группа_населения_возраст_детей"] = (pd.to_numeric(df["группа_населения_возраст_детей"], errors="coerce").astype("Int64"))
df["происхождение"] = pd.to_numeric(df["происхождение"], errors="coerce").astype("Int8")
df["количество_единиц_на_прием"] = pd.to_numeric(df["количество_единиц_на_прием"]).astype("Int64")
df["количество_приемов_в_день"] = pd.to_numeric(df["количество_приемов_в_день"]).astype("Int64")
binary_cols = [
    "пищевые_вещества_витамины_витаминоподобные_вещества_и_коферменты",
    "пищевые_вещества_макро_и_микроэлементы",
    "пищевые_вещества_белки_пептиды_аминокислоты_нуклеиновые_кислоты",
    "минорные_компоненты_растений_фенольные_соединения",
    "минорные_компоненты_растений_алкалоиды",
    "пробиотики_в_монокультурах_и_ассоциациях_пробиотические_микроорганизмы",
    "пищевые_вещества_углеводы_и_продукты_их_переработки",
    "минорные_компоненты_растений_сапонины",
    "минорные_компоненты_растений_терпеноиды",
    "минорные_компоненты_растений_естественные_метаболиты_и_стимуляторы_метаболизма",
    "минорные_компоненты_растений_гидроксикоричные_кислоты",
    "минорные_компоненты_растений_ферменты",
    "минорные_компоненты_растений_дубильные_вещества",
    "минеральные_и_минерало_органические_природные_субстанции_цеолиты_и_гуминовые_кислоты",
    "система_органов_для_беременных_кормящих_и_планирующих_беременность",
    "система_органов_костно_мышечная_система",
    "система_органов_нервная_система",
    "система_органов_иммунная_система",
    "система_органов_пищеварительный_тракт_и_обмен_веществ",
    "система_органов_мочеполовая_система",
    "система_органов_дерматологические_бад",
    "система_органов_органы_чувств",
    "система_органов_сердечно_сосудистая_система",
    "система_органов_противоопухолевые_бад",
    "система_органов_противопаразитарные_бад",
    "система_органов_кровь_и_система_кроветворения",
    "группа_населения_предназначен_для_детей",
    "группа_населения_предназначен_для_взрослых",
    "группа_населения_пожилые",
]
for col in binary_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int8")

# %% [markdown]
# Посчитаем столбец "суммарное_количество_единиц_за_период"

# %%
df['суммарное_количество_единиц_за_период'] = (
    df['количество_единиц_на_прием'] * df['количество_приемов_в_день'] * df['продолжительность_приема'] * 30
)

# %% [markdown]
# Посмотрим корреляцию числовых признаков

# %%
numeric_columns = df.select_dtypes(include=['int8', 'Int8', 'int64', 'Int64', 'float64', 'Float64']).columns
numeric_df = df[numeric_columns].copy()

correlation_pearson = numeric_df.corr(method='pearson') # linear data
correlation_spearman = numeric_df.corr(method='spearman') # unlinear data


# %% [markdown]
# Определим количество компонентов в столбце "биологически_активные_вещества" и занесем эти данные в столбец "количество_групп_компонентов"

# %%
def count_items(text):
    if pd.isna(text):
        return 0
    items = str(text).split(',')
    return len([item for item in items if item.strip()])

df_save['количество_групп_компонентов'] = df_save['биологически_активные_вещества'].apply(count_items)
df['количество_групп_компонентов'] = df_save['количество_групп_компонентов']
df_save = df_save.drop('количество_групп_компонентов', axis=1)

# %% [markdown]
# Heatmap на основе корреляции

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# Heatmap Пирсон
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_pearson,
            annot=False,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=0.5)
plt.title('Корреляция Пирсона (линейная)')
plt.tight_layout()
plt.show()

# Heatmap Спирман
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_spearman,
            annot=False,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=0.5)
plt.title('Корреляция Спирмана (ранговая)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Выберем пары с самой высокой корреляцией

# %%
correlation_matrix = df[numeric_columns].corr()

strong_pairs = []
for i in range(len(correlation_matrix.columns)):
  for j in range(i+1, len(correlation_matrix.columns)):
    corr = correlation_matrix.iloc[i, j]
    if abs(corr) >= 0.2 and (j != "количество_приемов_в_день"):
      strong_pairs.append({
          'feature1': correlation_matrix.columns[i],
          'feature2': correlation_matrix.columns[j],
          'correlation': corr
      })

# Удаляем бессмысленные
significant_pairs = sorted(strong_pairs, key= lambda x: abs(x['correlation']), reverse=True)[:-1]

significant_pairs = [
    p for p in significant_pairs
    if not ('группа_населения_возраст_детей' in p['feature1']
            and 'группа_населения_предназначен_для_взрослых' in p['feature2'])
    and not ('группа_населения_возраст_детей' in p['feature2']
            and 'группа_населения_предназначен_для_взрослых' in p['feature1'])
]


# %% [markdown]
# Построим диаграмму рассеивания для пар с самой высокой корреляцией

# %%
from scipy.stats import pearsonr

if significant_pairs:
    n_cols = 3
    n_rows = (len(significant_pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))

    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    # Цветовая палитра для кластеров
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    for idx, pair in enumerate(significant_pairs):
        if idx < len(axes):
            ax = axes[idx]
            feature1 = pair['feature1']
            feature2 = pair['feature2']
            corr_value = pair['correlation']

            # Данные для текущей пары (удаляем пропуски)
            data_pair = numeric_df[[feature1, feature2]].dropna()
            x_data = data_pair[feature1]
            y_data = data_pair[feature2]

            # Если оба признака бинарные
            if x_data.nunique() <= 2 and y_data.nunique() <= 2:
                # Создаем группы для каждого сочетания бинарных значений
                groups = data_pair.groupby([feature1, feature2]).size().reset_index(name='count')

                # Рисуем точки для каждой группы с разным цветом
                for i, (_, group) in enumerate(groups.iterrows()):
                    x_val = group[feature1]
                    y_val = group[feature2]
                    count = group['count']

                    # Добавляем jitter для разделения точек
                    jitter_x = np.random.normal(0, 0.03, count)
                    jitter_y = np.random.normal(0, 0.03, count)

                    ax.scatter(x_val + jitter_x, y_val + jitter_y,
                              alpha=0.7, s=50, color=colors[i % len(colors)],
                              label=f'({x_val},{y_val}): {count}')

                # Добавляем подписи количества для каждой группы
                for _, group in groups.iterrows():
                    x_val = group[feature1]
                    y_val = group[feature2]
                    count = group['count']

                    # Подписываем количество над кластером
                    ax.text(x_val, y_val + 0.1, f'n={count}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                # Легенда
                ax.legend(loc='upper left', bbox_to_anchor=(0, 0.5), fontsize=8)

                # 6. Настройки осей
                ax.set_xlabel(f'{feature1}\n(уник: {x_data.nunique()})', fontsize=11)
                ax.set_ylabel(f'{feature2}\n(уник: {y_data.nunique()})', fontsize=11)

            else:
                scatter = ax.scatter(x_data, y_data, alpha=0.5, s=40, c='steelblue')

                # 2. Линия тренда
                if len(x_data) > 1:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8,
                           label=f'Тренд: y={z[0]:.3f}x+{z[1]:.3f}')

                # 3. Вычисляем дополнительную статистику
                stats_text = f"""N = {len(data_pair)}
Корреляция = {corr_value:.3f}
p-value = {pearsonr(x_data, y_data)[1]:.4f}
R² = {corr_value**2:.3f}
x: μ={x_data.mean():.2f} σ={x_data.std():.2f}
y: μ={y_data.mean():.2f} σ={y_data.std():.2f}"""

                # 4. Добавляем текстовый блок со статистикой
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                       facecolor='lightyellow', alpha=0.8))

                # 6. Настройки осей
                ax.set_xlabel(f'{feature1}\n(уник: {x_data.nunique()})', fontsize=11)
                ax.set_ylabel(f'{feature2}\n(уник: {y_data.nunique()})', fontsize=11)

                # 7. Заголовок с информацией о типе связи
                if abs(corr_value) > 0.7:
                    strength = "ОЧЕНЬ СИЛЬНАЯ"
                elif abs(corr_value) > 0.5:
                    strength = "СИЛЬНАЯ"
                elif abs(corr_value) > 0.2:
                    strength = "УМЕРЕННАЯ"
                else:
                    strength = "СЛАБАЯ"

                direction = "ПОЛОЖИТЕЛЬНАЯ" if corr_value > 0 else "ОТРИЦАТЕЛЬНАЯ"

                # 8. Сетка
                ax.grid(True, alpha=0.3)

                # 9. Легенда
                ax.legend(loc='lower right', fontsize=9)

    # Скрываем пустые subplots
    for idx in range(len(significant_pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# Объединение столбцов:биологически_активные_вещества, системы_органы,группы_населения с основынм датафреймом. Удаляем столбцы, которые были использованы при объединении

# %%
df = df.join(df_save)
for col in biolog_columns:
  df = df.drop(col, axis=1)
for col in sys_org:
  df = df.drop(col, axis=1)
for col in group_people:
  df = df.drop(col, axis=1)

# %% [markdown]
# Добавим парсинг столбца сырье на ингредиент_описание, ингредиент_рус, ингредиент_лат

# %% [markdown]
# Функция для парсинга строки с сырьем на отдельные ингредиенты(если много ингредиентов через запятую)

# %%
import re

# Парсим строку с сырьем
def parse_ingredient_string(raw_string):
  if pd.isna(raw_string) or not raw_string.strip():
    return []

  ingredients = []
  current = []
  bracket_count = 0
  quote_count = 0

  for char in raw_string:
    if char == '(':
      bracket_count += 1
    elif char == ')':
      bracket_count -= 1
    elif char == '"':
      quote_count = 1 - quote_count

    if char == ',' and bracket_count == 0 and quote_count == 0:
      ingredient_str = ''.join(current).strip()
      if ingredient_str:
        ingredients.append(ingredient_str)
      current = []
    else:
      current.append(char)

  if current:
    ingredient_str = ''.join(current).strip()
    if ingredient_str:
      ingredients.append(ingredient_str)

  return ingredients


# %% [markdown]
# Функция точно определяет, где русское название, а где латинское

# %%
def detect_language(text):
    cyrillic_count = len(re.findall(r'[а-яА-Я]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    if cyrillic_count > latin_count:
        return 'russian'
    elif latin_count > cyrillic_count:
        return 'latin'
    else:
        # Если равное количество символов или оба нуля, используем дополнительные признаки
        if re.search(r'[а-яА-Я]', text):
            return 'russian'
        elif re.search(r'[a-zA-Z]', text):
            return 'latin'
        return 'unknown'


# %% [markdown]
# Функция для парсинга строки с сырьем на отдельные ингредиенты(если много ингредиентов через запятую)

# %%
def parse_single_ingredient(ingredient):
    ingredient = str(ingredient).strip().lower()

    if ingredient in ['nan', 'None', '']:
        return (pd.NA, pd.NA, pd.NA)

    if '(' in ingredient and ')' not in ingredient:
        ingredient = ingredient + ')'
    pattern1 = r'^(.+?)\s*\(([^()]+?)\s*[–\-—]\s*([^()]+?)\)\s*\.?$'
    match1 = re.match(pattern1, ingredient)

    if 'содержит бактерии' in ingredient:
      return (ingredient, pd.NA, pd.NA)
    if match1:
        description = match1.group(1).strip()
        first_part = match1.group(2).strip()
        second_part = match1.group(3).strip()

        description = description.rstrip('.')
        first_part = first_part.rstrip('.').strip('–').strip()
        second_part = second_part.rstrip('.').strip('–').strip()

        # Определяем язык для каждой части
        lang_first = detect_language(first_part)
        lang_second = detect_language(second_part)

        if lang_first == 'russian' and lang_second == 'latin':
            return (description, first_part, second_part)
        elif lang_first == 'latin' and lang_second == 'russian':
            if bool(re.search(r'[a-zA-Z]', second_part)) and ingredient.count("-") == 2:
              first_part += "-" + second_part[:second_part.find("-")]
              second_part = second_part[second_part.find("-")+1:]
            return (description, second_part, first_part)

        return (description, second_part, first_part)

    pattern2 = r'^(.+?)\s*\(([^()]+?)\)\s*\.?$'
    match2 = re.match(pattern2, ingredient)

    if match2:
        description = match2.group(1).strip()
        content = match2.group(2).strip()

        has_latin = bool(re.search(r'[a-zA-Z]', content))
        has_cyrillic = bool(re.search(r'[а-яА-Я]', content))

        description = description.rstrip('.')
        content = content.rstrip('.').strip('–').strip()

        if has_latin and not has_cyrillic:
          return (description, pd.NA, content)
        elif has_cyrillic and not has_latin:
            return (description, content, pd.NA)
        else:
            return (description, content, pd.NA)

    ingredient = ingredient.rstrip('.')
    return (ingredient, pd.NA, pd.NA)


# %%
df["ингредиент_описание"] = pd.NA
df["ингредиент_рус"] = pd.NA
df["ингредиент_лат"] = pd.NA

for row in range(len(df)):
    raw_value = df.at[row, "сырье"]

    if pd.isna(raw_value) or str(raw_value).strip() in ['', 'nan', 'None']:
        df.at[row, "ингредиент_описание"] = pd.NA
        df.at[row, "ингредиент_рус"] = pd.NA
        df.at[row, "ингредиент_лат"] = pd.NA
        continue

    string = str(raw_value).strip().lower()
    ingredients = parse_ingredient_string(string)

    if not ingredients:
        description, russian, latin = parse_single_ingredient(string)
        df.at[row, "ингредиент_описание"] = description
        df.at[row, "ингредиент_рус"] = russian
        df.at[row, "ингредиент_лат"] = latin

    else:
        description_list = []
        russian_list = []
        latin_list = []

        for ingredient in ingredients:
            description, russian, latin = parse_single_ingredient(ingredient)
            if description:
                description_list.append(description)
            if pd.notna(russian):
                russian_list.append(russian)
            if pd.notna(latin):
                latin_list.append(latin)

        df.at[row, "ингредиент_описание"] = ", ".join(description_list) if description_list else pd.NA
        df.at[row, "ингредиент_рус"] = ", ".join(russian_list) if russian_list else pd.NA
        df.at[row, "ингредиент_лат"] = ", ".join(latin_list) if latin_list else pd.NA

# %% [markdown]
# Удалим строки, у которых природное происхождение, но отсутствуют данные о сырье

# %%
indx_for_drop = []
for i in range(len(df)):
  if "Синтетическое" not in str(df.at[i, "происхождение_природное_синтетическое"]) and pd.isna(df.at[i, "сырье"]):
    indx_for_drop.append(i)

df = df.drop(indx_for_drop)


# %% [markdown]
# Функция для извлечения ингредиентов из строки описания

# %%
def extract_ingredients(description):
    if pd.isna(description) or description == "":
        return []

    # Удаляем содержимое в скобках
    description_clean = re.sub(r'\([^)]*\)', '', str(description)).lower()

    # Разделяем по запятым и очищаем от лишних пробелов
    ingredients = [ing.strip().lower() for ing in description_clean.split(',')]

    # Удаляем пустые строки и строки, содержащие только знаки препинания
    ingredients = [ing for ing in ingredients if ing and ing.strip()]

    return ingredients


# %% [markdown]
# Удаление дополнительных кавычек и точек

# %%
def clean_ingredient_name(ingredient):
    # Удаляем кавычки в начале и конце
    ingredient = re.sub(r'^["\']|["\']$', '', ingredient)
    # Удаляем точки в конце
    ingredient = re.sub(r'\.$', '', ingredient)
    # Удаляем лишние пробелы
    ingredient = ingredient.strip()
    return ingredient


# %% [markdown]
# Извлекаем все ингредиенты из описания
#

# %%
df['ингредиенты_список'] = df['ингредиент_описание'].apply(extract_ingredients)
df['ингредиенты_список'] = df['ингредиенты_список'].apply(
    lambda x: [clean_ingredient_name(ing) for ing in x if clean_ingredient_name(ing)]
)


# %%
all_ingredients = set()
for ingredient_list in df['ингредиенты_список']:
  all_ingredients.update(ingredient_list)

all_ingredients = sorted(list(all_ingredients))
print(f"Всего уникальных значений: {len(all_ingredients)}")
all_ingredients.remove("\\")
print("Игредиенты:", all_ingredients[:100])

# %% [markdown]
# Создаем матрицу ингредиентов и корреляций

# %%
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

mlb = MultiLabelBinarizer()
ingredient_matrix = mlb.fit_transform(df['ингредиенты_список'])
ingredient_df = pd.DataFrame(ingredient_matrix, columns=mlb.classes_)

# Только ингредиенты, встречающиеся >= 10 раз
frequent_ingredients = ingredient_df.columns[ingredient_df.sum() >= 10]
filtered_df = ingredient_df[frequent_ingredients]

# Корреляция и визуализация
corr_matrix = pd.DataFrame(
    cosine_similarity(filtered_df.T),
    index=frequent_ingredients,
    columns=frequent_ingredients
)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Корреляции между ингредиентами (встречаются ≥10 раз)')
plt.show()

print(f"Проанализировано {len(frequent_ingredients)} ингредиентов")

# %% [markdown]
# Перейдем к построению графов. Начнём с графов двойной композиции.
#
# В первом случае у нас:
# - вершины - биологически активные вещества
# - ребра - наличие БАД содержащих эти биологически активные вещества
# - вес - количество БАД содержащих эти биологически активные вещества
#
# Во втором случае у нас:
# - вершины - растения
# - ребра - наличие БАД содержащих эти растения
# - вес - количество БАД содержащих эти растения

# %% [markdown]
# Напишем функцию, которая находит ребра и веса этих графов, функцию, которая разбирает строку в список элементов, а также функцию которая убирает значения меньше определенного в словаре

# %%
from itertools import combinations

def parse_items(cell, sep=",", mapper=None):
    if pd.isna(cell):
        return []

    raw_items = [x.strip() for x in str(cell).split(sep)]

    items = []
    for x in raw_items:
        if not x:
            continue
        if mapper is not None:
            x = mapper(x)
        items.append(x)

    return sorted(set(items))

def count_pairs(df, col, sep=",", mapper=None):
    pair_counts = {}

    for cell in df[col]:
        items = parse_items(cell, sep=sep, mapper=mapper)
        if len(items) < 2:
            continue

        for a, b in combinations(items, 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    return pair_counts

def filter_dictionary_by_value(dict, threshold):
    filtered_dict = {
        key: value
        for key, value in dict.items()
        if value >= threshold
    }
    return filtered_dict


# %% [markdown]
# Поскольку нам важно, чтобы на графе были корректные и полнообъемные названия, то создадим словарь замен

# %%
CLASS_MAP = {
    "вит": "Витамины, витаминоподобные вещества и коферменты",
    "элементы": "Макро- и микроэлементы",
    "пнжк": "Жиры, жироподобные вещества и их производные",
    "стеран": "Жиры, жироподобные вещества и их производные",
    "аминокислоты": "Белки, пептиды, аминокислоты, нуклеиновые кислоты",
    "фенольн": "Фенольные соединения",
    "алкалоиды": "Алкалоиды",
    "пробиотики": "Пробиотические микроорганизмы",
    "полисахариды": "Углеводы и продукты их переработки",
    "сапонины": "Сапонины",
    "терпен": "Терпеноиды",
    "ест": "Естественные метаболиты и стимуляторы метаболизма",
    "гидроксикор": "Гидроксикоричные кислоты",
    "ферменты": "Ферменты",
    "дуб": "Дубильные вещества",
    "цеолиты": "Цеолиты и гуминовые кислоты",
}

# %% [markdown]
# Теперь установим и импортируем нужные библиотеки для построения графов

# %%
# !pip install pyvis
# !pip install --upgrade pyvis
from pyvis.network import Network
from jinja2 import Environment, FileSystemLoader
import pyvis
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
import math
import colorsys


# %% [markdown]
# Напишем функцию, создающую html страницу с интерактивным графом

# %%
def create_interactive_graph(pairs, output_path="interactive_graph.html"):
    if not pairs:
        print("Словарь пар пуст")
        return

    min_w = min(pairs.values())
    max_w = max(pairs.values())

    nodes_set = set()
    for (u, v) in pairs.keys():
        nodes_set.add(u)
        nodes_set.add(v)

    num_nodes = len(nodes_set)
    num_edges = len(pairs)

    nodes_sorted = sorted(nodes_set)
    node_colors = {}

    if num_nodes > 0:
        for idx, node in enumerate(nodes_sorted):
            hue = idx / float(num_nodes)
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.8)
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(r * 255),
                int(g * 255),
                int(b * 255),
            )
            node_colors[node] = color_hex

    net = Network(
        height="1000px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        cdn_resources="in_line",
    )

    max_w_for_width = max_w if max_w > 0 else 1.0

    for (u, v), w in pairs.items():

        norm = max(w, 0) / max_w_for_width
        width = 2.0 + 3.0 * math.sqrt(norm)

        if u not in node_colors:
            node_colors[u] = "#8ab4f8"
        if v not in node_colors:
            node_colors[v] = "#8ab4f8"

        net.add_node(u, label=u, color=node_colors[u])
        net.add_node(v, label=v, color=node_colors[v])

        net.add_edge(
            u,
            v,
            title=f"совместно в {w} БАДах",
            width=width,
            value=w,
            color={"inherit": "both"},
        )

    options = {
        "interaction": {
            "hover": True,
            "hoverConnectedEdges": True,
            "selectConnectedEdges": True,
        },
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 10,
                "max": 30,
            },
            "font": {
                "size": 16,
            },
        },
        "edges": {
            "smooth": {
                "enabled": True,
                "type": "dynamic",
                "roundness": 0.4,
            },
            "color": {
                "inherit": "both",
            },
        },
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -30000,
                "centralGravity": 0.01,
                "springLength": 350,
                "springConstant": 0.01,
                "damping": 0.09,
                "avoidOverlap": 0.7,
            },
            "stabilization": {
                "iterations": 300,
            },
        },
    }

    net.set_options(json.dumps(options))

    templates_path = os.path.join(os.path.dirname(pyvis.__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_path))
    net.template = env.get_template("template.html")

    net.write_html(output_path)
    add_weight_form_to_html(
        output_path,
        min_w=min_w,
        max_w=max_w,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )

    print("HTML сохранён в", output_path)

def add_weight_form_to_html(html_path, min_w, max_w, num_nodes, num_edges):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    placeholder = '<div id="mynetwork" class="card-body"></div>'

    controls_html = f"""
<div id="top-panel"
     style="
        position:fixed;
        top:0; left:0; right:0;
        z-index:9999;
        background:rgba(20,20,20,0.95);
        border-bottom:1px solid #444;
        padding:8px 20px;
        display:flex;
        align-items:center;
        gap:16px;
        font-family:Arial, sans-serif;
        font-size:14px;
        color:#eee;
     ">
  <div style="font-weight:600; white-space:nowrap;">
    Граф сочетаемости БАД-классов
  </div>
  <div style="opacity:0.8; white-space:nowrap;">
    Вершин: {num_nodes} · Рёбер: {num_edges} · Макс. вес ребра: {max_w}
  </div>
  <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
    <label style="white-space:nowrap;">
      Порог веса:
      <input type="number"
             id="minWeightInput"
             value="{min_w}"
             min="{min_w}"
             max="{max_w}"
             step="1"
             style="
                width:70px;
                margin-left:4px;
                padding:2px 4px;
                background:#111;
                border:1px solid #555;
                color:#eee;
                border-radius:4px;
             ">
    </label>
    <button id="applyWeightBtn"
            style="
                padding:3px 12px;
                border-radius:4px;
                border:1px solid #666;
                background:#2d6cdf;
                color:#fff;
                cursor:pointer;
            ">
      Применить
    </button>
    <span id="minWeightInfo"
          style="margin-left:4px; font-size:12px; opacity:0.85;">
      Порог: ≥ {min_w}
    </span>
  </div>
</div>
<!-- отступ, чтобы граф не залез под фиксированную панель -->
<div style="height:48px;"></div>
""".strip()

    if placeholder in html:
        html = html.replace(placeholder, controls_html + "\n\n" + placeholder, 1)
    else:
        print("Не найден div с id='mynetwork' class='card-body' — панель не вставлена")

    js_block = f"""
<script type="text/javascript">
window.addEventListener("load", function () {{
    if (typeof edges === "undefined") {{
        console.warn("edges DataSet not found");
        return;
    }}

    var allEdges = edges.get();
    var input = document.getElementById("minWeightInput");
    var btn   = document.getElementById("applyWeightBtn");
    var info  = document.getElementById("minWeightInfo");

    if (!input || !btn) {{
        console.warn("weight controls not found");
        return;
    }}

    function applyThreshold() {{
        var v = parseInt(input.value);
        if (isNaN(v)) {{
            v = {min_w};
            input.value = v;
        }}
        if (info) {{
            info.textContent = "Порог: ≥ " + v;
        }}

        var updates = [];
        for (var i = 0; i < allEdges.length; i++) {{
            var e = allEdges[i];
            var hide = e.value < v;
            updates.push({{id: e.id, hidden: hide}});
        }}
        edges.update(updates);
    }}

    btn.addEventListener("click", applyThreshold);
    input.addEventListener("keyup", function(e) {{
        if (e.key === "Enter") {{
            applyThreshold();
        }}
    }});

    applyThreshold();
}});
</script>
"""

    html = html.replace("</body>", js_block + "\n</body>", 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)



# %%
def export_graph_png(
    pairs,
    output_path="bad_graph_colored.png",
    min_weight=1,
    label_min_weight=1,
    cmap_name="viridis",
    curve_scale=0.25,
    label_t_ranges=((0.2, 0.4), (0.6, 0.8)),  # зоны для лейблов
):
    num_segments = 20

    G = nx.Graph()
    for (u, v), w in pairs.items():
        if w < min_weight:
            continue
        G.add_edge(u, v, weight=w)

    if G.number_of_edges() == 0:
        print("После порога нет рёбер, PNG не из чего рисовать")
        return

    pos_nodes = {n: np.array(p) for n, p in nx.circular_layout(G).items()}

    fig, ax = plt.subplots(figsize=(16, 12))

    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}

    base_cmap = mpl.colormaps.get_cmap(cmap_name)
    colors_for_nodes = base_cmap(np.linspace(0, 1, max(n_nodes, 1)))

    node_color_dict = {
        node: colors_for_nodes[i % colors_for_nodes.shape[0]]
        for i, node in enumerate(nodes)
    }
    node_colors = [node_color_dict[n] for n in nodes]

    weights = [w for (_, _, w) in G.edges(data="weight")]
    max_w = max(weights) if weights else 1.0

    segments = []
    segment_colors = []
    segment_widths = []
    label_infos = []

    center = np.array([0.0, 0.0])

    for (u, v, w) in G.edges(data="weight"):
        p0 = np.array(pos_nodes[u])
        p1 = np.array(pos_nodes[v])

        dir_vec = p1 - p0
        dist = np.linalg.norm(dir_vec)
        if dist == 0:
            continue

        mid = 0.5 * (p0 + p1)

        perp = np.array([-dir_vec[1], dir_vec[0]]) / dist
        c1 = mid + perp * curve_scale * dist
        c2 = mid - perp * curve_scale * dist

        control = c1 if np.linalg.norm(c1 - center) < np.linalg.norm(c2 - center) else c2

        t_vals = np.linspace(0.0, 1.0, num_segments + 1)
        a = ((1 - t_vals) ** 2)[:, None]
        b = (2 * (1 - t_vals) * t_vals)[:, None]
        c = (t_vals ** 2)[:, None]
        points = a * p0 + b * control + c * p1

        cu = np.array(node_color_dict[u])
        cv = np.array(node_color_dict[v])

        edge_width = 1.0 + 4.0 * (w / max_w)

        for i in range(num_segments):
            p_start = points[i]
            p_end = points[i + 1]
            segments.append([p_start, p_end])

            t_mid = (t_vals[i] + t_vals[i + 1]) / 2.0
            c_mid = cu * (1 - t_mid) + cv * t_mid
            segment_colors.append(c_mid)
            segment_widths.append(edge_width)

        iu, iv = node_index[u], node_index[v]
        r_idx = (iu + iv) % len(label_t_ranges)
        t_lo, t_hi = label_t_ranges[r_idx]
        t_label = 0.5 * (t_lo + t_hi)

        aL = (1 - t_label) ** 2
        bL = 2 * (1 - t_label) * t_label
        cL = t_label ** 2
        label_point = aL * p0 + bL * control + cL * p1
        mx, my = label_point

        label_color = cu * (1 - t_label) + cv * t_label
        label_infos.append((mx, my, w, label_color))

    lc = LineCollection(
        segments,
        colors=segment_colors,
        linewidths=segment_widths,
        alpha=0.9,
        capstyle="round",
        joinstyle="round",
    )
    lc.set_zorder(1)
    ax.add_collection(lc)

    node_collection = nx.draw_networkx_nodes(
        G,
        pos_nodes,
        node_size=900,
        node_color=node_colors,
        ax=ax,
    )
    node_collection.set_zorder(2)

    label_dict = nx.draw_networkx_labels(
        G,
        pos_nodes,
        font_size=10,
        ax=ax,
    )
    for text in label_dict.values():
        text.set_clip_on(False)
        text.set_zorder(3)

    for (mx, my, w, label_color) in label_infos:
        if w < label_min_weight:
            continue

        txt = ax.text(
            mx,
            my,
            str(w),
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=label_color,
                ec="none",
                alpha=0.9,
            ),
            color="black",
        )
        txt.set_clip_on(False)
        txt.set_zorder(3)

    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("PNG сохранён в", output_path)


# %%
pairs_of_components = count_pairs(df, "биологически_активные_вещества", mapper=lambda x: CLASS_MAP.get(x, x))
print(pairs_of_components)
pairs_of_raw = count_pairs(df, "ингредиент_описание")
print(pairs_of_raw)

# %% [markdown]
# Среди пар ингредиентов очень много ребер с весом 1. Они крайне неинформативны и более того мешающие. Поэтому уберем все ребра, которые меньше заданного значения веса

# %%
# pairs_of_raw = filter_dictionary_by_value(pairs_of_raw, 8)

# %%
create_interactive_graph(pairs_of_components, output_path="interactive_graph_of_components.html")
create_interactive_graph(pairs_of_raw, output_path="interactive_graph_of_raw.html")

# %%
export_graph_png(pairs_of_components, output_path="static_graph_of_components.png", min_weight=1, label_min_weight=1, cmap_name="tab20",
    curve_scale=0.25,
    label_t_ranges=((0.3, 0.45), (0.55, 0.7)),
)
export_graph_png(pairs_of_raw, output_path="static_graph_of_raw.png", min_weight=8, label_min_weight=1, cmap_name="tab20",
    curve_scale=0.25,
    label_t_ranges=((0.3, 0.45), (0.55, 0.7)),
)

# %% [markdown]
# # Сохранение изменений

# %%
import os, sys
import pathlib
from pathlib import Path

try:
    _system = get_ipython().system
except NameError:
    import subprocess
    def _system(cmd):
        return subprocess.check_call(cmd, shell=True)

IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not IN_COLAB:
    print("Окружение не Colab. Настройка на безопасную версию")
    NOTEBOOK = "3311_bajmuhamedov_arshin_pasechny_practice_bad.ipynb"
    cfg_path = ".jupytext.toml"
else:
    print("Running in Colab, executing Jupytext sync logic...")

    from google.colab import drive
    drive.mount('/content/drive')

    NOTEBOOK = "/content/drive/MyDrive/Colab Notebooks/3311_bajmuhamedov_arshin_pasechny_practice_bad.ipynb"
    cfg_path = "/content/.jupytext.toml"

_system(f'pip -q install jupytext nbstripout')
_system(f'nbstripout "{NOTEBOOK}"')

cfg = '''formats = "ipynb,py:percent"
cell_metadata_filter = "-all,tags"
notebook_metadata_filter = "kernelspec,jupytext"
'''
with open(cfg_path, "w", encoding="utf-8") as f:
    f.write(cfg)

if IN_COLAB:
    ipynb_path = Path(NOTEBOOK)
    py_path = ipynb_path.with_suffix(".py")

    if not ipynb_path.exists():
        raise FileNotFoundError(f"Не найден .ipynb: {ipynb_path}")

    print("IPYNB:", ipynb_path)
    print("PY:", py_path)

    _system(f'nbstripout "{NOTEBOOK}"')

    if py_path.exists():
        py_path.unlink()

    _system(f'jupytext --to py:percent "{NOTEBOOK}"')

    import datetime
    stat = py_path.stat()
    print("\nОбновлён .py:", py_path)

else:
    pass
