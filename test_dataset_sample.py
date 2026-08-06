#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""بناء عينة صغيرة من البيانات للاختبار (100 رابط فقط)"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

import pandas as pd
from src.feature_extractor import FEATURE_NAMES, extract_features, get_domain

# قراءة 100 رابط فقط من المصدر
df = pd.read_csv("data/raw/Final_Phishing_Dataset_96k.csv", 
                 usecols=["url", "label"], 
                 nrows=100)

print(f"قراءة {len(df)} رابط من المصدر\n")
print("استخراج الميزات مع network='auto'...")

rows = []
for i, url in enumerate(df["url"], start=1):
    if i % 20 == 0:
        print(f"  {i}/100...")
    rows.append(extract_features(url, network="auto"))

result = pd.DataFrame(rows, columns=FEATURE_NAMES)
result.insert(0, "url", df["url"].values)
result.insert(1, "domain", [get_domain(u) for u in df["url"]])
result.insert(2, "label", df["label"].values)

output_path = "data/processed/sample_100.csv"
result.to_csv(output_path, index=False)

print(f"\n✓ تم الحفظ في: {output_path}")
print(f"  الأعمدة: {len(result.columns)}")
print(f"  الصفوف: {len(result)}")
print(f"\nعينة من الأعمدة:")
print(list(result.columns)[:10])
