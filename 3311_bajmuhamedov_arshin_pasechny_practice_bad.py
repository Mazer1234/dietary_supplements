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
#       jupytext_version: 1.17.3
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
# Посмотрим суммарную  информацию о датафрейме

# %%
print_info(df)

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

    # Преобразуем столбцы с двумя уникальными значениями в бинарные
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
    ["группа_населения_пожилые", ["пожилые"], "1"],
]

for i in range(len(pairs)):
    replace_exact(df, pairs[i][0],pairs[i][1], pairs[i][2])

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
# Посмотрим суммарную ифнормацию о получившемся датафрейме

# %%
print_info(df)

# %% [markdown]
# Посмотрим как выглядит датафрейм на данный момент

# %%
df.head()

# %% [markdown]
# Вопросы на рассмотрение:
# Порог процента пустых значений, при котором мы отбросим столбец

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
# Посмотрим как сейчас выглядит датафрейм

# %%
df.head()

# %% [markdown]
# Выведем суммарную информацци о датафрейме

# %%
print_info(df)

# %% [markdown]
# # Сохранение изменений

# %% tags=["skip"]
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
    print("Environment is NOT Colab. Setting safe variables.")
    
    NOTEBOOK = "dummy_notebook.ipynb"
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

    if py_path.exists():
        py_path.unlink()
    
    _system(f'jupytext --to py:percent "{NOTEBOOK}"') 

    import datetime
    stat = py_path.stat()
    print("\nОбновлён .py:", py_path)
    
else:
    pass
