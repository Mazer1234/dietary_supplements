import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode
from pathlib import Path
import re
from collections import Counter, defaultdict
base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = "https://disk.yandex.ru/d/V1sJpR-SUJ_b8A"

final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

download_response = requests.get(download_url)
with open('dataset.xlsx', 'wb') as f:
    f.write(download_response.content)
#------------------------------------------------------------------
xlsx_path = "dataset.xlsx"
if not xlsx_path:
    raise FileNotFoundError("xlsx файл не найден")
print("Найден XLSX:", xlsx_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#------------------------------------------------------------------
df = pd.read_excel(xlsx_path, sheet_name=0, header=[0,1])
print("Данные загружены в df")

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
#------------------------------------------------------------------


#------------------------------------------------------------------
#Приведение столбцов к бинарным
#------------------------------------------------------------------
df_copy= df.copy(deep=True)

df_copy['происхождение'] = df_copy['происхождение'].map({'отечественное': 1, 'иностранное': 0})
#------------------------------------------------------------------
forms_release = [
    'таблетки', 'капсулы', 'растворы', 'порошок', 'сбор', 
    'пастилки', 'драже', 'пилюли', 'гели',
    'гранулы','желе','леденцы','пасты','плитки','суспензия'
]

df_copy['форма_выпуска'] = (
    df_copy['форма_выпуска']
    .astype(str)
    .str.lower()
    .str.replace(r'[^а-яa-z, ]', '', regex=True)
)

for form in forms_release:
    df_copy[form] = df_copy['форма_выпуска'].apply(
        lambda x: 1 if form in x else 0
    )

# Проверим результат
print("Бинарные признаки добавлены в Формы выпуска:")
print(df_copy[[*forms_release]].head())


# ------------------------------------------------
# Продолжительность приема
# ------------------------------------------------



# ------------------------------------------------
# Происхождение (природное/синтетическое)
# ------------------------------------------------
