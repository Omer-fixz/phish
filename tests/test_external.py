# =========================================================
# اختبار خارجي على روابط تصيد حديثة
# =========================================================
# لماذا هذا الاختبار؟
# كل الأرقام السابقة (دقة 0.90، ROC-AUC 0.97) قيست على مجموعة اختبار
# مأخوذة من نفس مصدر بيانات التدريب. هذا يثبت أن النموذج تعلّم، لكنه
# لا يثبت أنه ينفع على تصيد اليوم. مصدر البيانات الأصلي قديم، وأساليب
# المهاجمين تغيّرت.
#
# لذلك نختبر هنا على مصدرين مستقلين تماماً لم يرهما النموذج إطلاقاً:
#   1. PhishTank  — روابط تصيد مؤكدة يدوياً من متطوعين، لا تزال حيّة.
#   2. OpenPhish  — تغذية مستقلة ثانية للتحقق المتقاطع.
# والجانب الشرعي من نطاقات تارانكو العميقة (ترتيب 200 ألف فأكثر)،
# وهي خارج الشريحة التي استُخدمت في بناء البيانات.
#
# قاعدة أمنية: الروابط هنا تصيد حيّ فعلاً. تُعامَل كنصوص فقط،
# ولا تُفتح ولا يُطلب محتواها في أي سطر من هذا الملف.
# =========================================================

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys, pathlib
# نضيف جذر المشروع إلى مسار البحث حتى نستطيع استيراد src من أي مجلد
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


import gzip                                # لفك ضغط ملف PhishTank
import json                                # لقراءة ترتيب الخصائص
import os                                  # للتعامل مع الملفات
import joblib                              # لتحميل النموذج
import numpy as np
import pandas as pd

from src.paths import (PHISHTANK_GZ, OPENPHISH_TXT, TRANCO_CSV, DATASET_CSV,
                       MODEL_FILE, FEATURES_JSON)
from src.feature_extractor import extract_features, get_domain

# ---------------------------------------------------------
# إعدادات
# ---------------------------------------------------------
PHISHTANK_FILE = PHISHTANK_GZ    # تغذية PhishTank
OPENPHISH_FILE = OPENPHISH_TXT   # تغذية OpenPhish
TRANCO_FILE = TRANCO_CSV         # قائمة المواقع الشهيرة
TRAIN_FILE = DATASET_CSV         # بيانات التدريب (لاستبعاد المكرر)

SAMPLE_PHISH = 3000                        # كم رابط تصيد نختبر
LEGIT_FROM, LEGIT_TO = 200_000, 400_000    # شريحة تارانكو العميقة (خارج التدريب تماماً)
SAMPLE_LEGIT = 3000                        # كم رابط شرعي نختبر
SEED = 42


# =========================================================
# 1. تحميل النموذج
# =========================================================
bundle = joblib.load(MODEL_FILE)
names = json.load(open(FEATURES_JSON, encoding="utf-8"))
model, threshold = bundle["model"], bundle["threshold"]

print("=" * 66)
print("اختبار خارجي على روابط تصيد حديثة")
print("=" * 66)
print(f"النموذج: {bundle['model_name']} | العتبة: {threshold} | الخصائص: {len(names)}")
print(f"دقته على بيانات المصدر الأصلي: {bundle['test_metrics']['accuracy']:.4f}")


def predict(urls):
    """يحوّل قائمة روابط إلى قرارات واحتمالات — بلا أي اتصال بالشبكة."""
    rows = [extract_features(u) for u in urls]              # الاستخراج بالمستخرج الرسمي
    X = pd.DataFrame(rows, columns=None)[names].values      # المتجهات بترتيب الخصائص الثابت
    proba = model.predict_proba(X)[:, 1]                    # احتمال التصيد
    return proba, (proba >= threshold).astype(int)          # القرار عند العتبة المعتمدة


# =========================================================
# 2. تحميل روابط التصيد الحديثة
# =========================================================
def load_phishtank():
    """يقرأ تغذية PhishTank ويأخذ الأحدث منها."""
    if not PHISHTANK_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(gzip.open(PHISHTANK_FILE), low_memory=False)
    df = df[df["verified"] == "yes"]                        # المؤكدة يدوياً فقط
    if "online" in df.columns:
        df = df[df["online"] == "yes"]                      # ولا تزال حيّة
    df = df.sort_values("submission_time", ascending=False)  # الأحدث أولاً
    print(f"\n📥 PhishTank: {len(df)} رابطاً مؤكداً وحيّاً")
    print(f"   أحدث إضافة: {df['submission_time'].iloc[0][:10]}")
    print(f"   أقدم إضافة: {df['submission_time'].iloc[-1][:10]}")
    return df


def load_openphish():
    """يقرأ تغذية OpenPhish (نص عادي، رابط في كل سطر)."""
    if not OPENPHISH_FILE.exists():
        return []
    with open(OPENPHISH_FILE, encoding="utf-8", errors="replace") as fh:
        urls = [ln.strip() for ln in fh if ln.strip().startswith("http")]
    print(f"📥 OpenPhish: {len(urls)} رابطاً")
    return urls


# =========================================================
# 3. تحميل الروابط الشرعية من شريحة لم تُستخدم في التدريب
# =========================================================
def load_legit():
    """
    يأخذ نطاقات من عمق قائمة تارانكو، خارج الشريحة المستخدمة في البناء.

    قيد مهم يجب ذكره: هذه نطاقات مجرّدة بلا مسار، وهي أسهل حالة على
    النموذج. لذلك نسبة الإنذارات الكاذبة المقاسة هنا متفائلة، ولا
    تُقارن مباشرة بنسبة الإنذارات على بيانات المصدر الأصلي.
    """
    tr = pd.read_csv(TRANCO_FILE, names=["rank", "domain"], skiprows=LEGIT_FROM,
                     nrows=LEGIT_TO - LEGIT_FROM)
    picked = tr.sample(SAMPLE_LEGIT, random_state=SEED)["domain"].tolist()
    print(f"📥 نطاقات شرعية من تارانكو (ترتيب {LEGIT_FROM:,}–{LEGIT_TO:,}): {len(picked)}")
    return picked


# =========================================================
# 4. تنفيذ التقييم
# =========================================================
pt = load_phishtank()
op = load_openphish()
legit = load_legit()

# نطاقات التدريب — لقياس الأداء على نطاقات لم يرها النموذج إطلاقاً
train_domains = set(pd.read_csv(TRAIN_FILE, usecols=["domain"])["domain"])
print(f"📚 نطاقات التدريب المعروفة: {len(train_domains):,}")

phish_urls = pt["url"].astype(str).head(SAMPLE_PHISH).tolist()   # عيّنة الأحدث

print("\n" + "=" * 66)
print("النتائج")
print("=" * 66)

# ---- الجانب الخبيث: PhishTank ------------------------------------------
proba_p, pred_p = predict(phish_urls)
recall_pt = pred_p.mean()                                   # نسبة ما أمسكه النموذج
print(f"\n🎣 PhishTank ({len(phish_urls)} رابطاً):")
print(f"   نسبة الكشف (Recall) = {recall_pt:.4f}  "
      f"أي أمسك {int(pred_p.sum())} وفوّت {int((1-pred_p).sum())}")

# الأداء على النطاقات الجديدة كلياً (الاختبار الأصعب والأصدق)
new_mask = np.array([get_domain(u) not in train_domains for u in phish_urls])
if new_mask.any():
    print(f"   منها {int(new_mask.sum())} رابطاً على نطاقات لم يرها النموذج: "
          f"كشف = {pred_p[new_mask].mean():.4f}")

# ---- الجانب الخبيث: OpenPhish (تحقق متقاطع) -----------------------------
if op:
    proba_o, pred_o = predict(op)
    print(f"\n🎣 OpenPhish ({len(op)} رابطاً) — مصدر مستقل ثانٍ:")
    print(f"   نسبة الكشف (Recall) = {pred_o.mean():.4f}")

# ---- الجانب الشرعي -------------------------------------------------------
proba_l, pred_l = predict(legit)
fpr = pred_l.mean()                                         # نسبة الإنذارات الكاذبة
print(f"\n✅ نطاقات شرعية ({len(legit)}):")
print(f"   إنذارات كاذبة (FPR) = {fpr:.4f}  أي {int(pred_l.sum())} نطاقاً حُجب خطأً")

# ---- الملخص العام --------------------------------------------------------
y_true = np.r_[np.ones(len(phish_urls)), np.zeros(len(legit))]
y_pred = np.r_[pred_p, pred_l]
acc = (y_true == y_pred).mean()
prec = pred_p.sum() / (pred_p.sum() + pred_l.sum()) if (pred_p.sum() + pred_l.sum()) else 0
f1 = 2 * prec * recall_pt / (prec + recall_pt) if (prec + recall_pt) else 0

print(f"\n📊 الإجمالي على البيانات الخارجية:")
print(f"   Accuracy  = {acc:.4f}")
print(f"   Precision = {prec:.4f}")
print(f"   Recall    = {recall_pt:.4f}")
print(f"   F1        = {f1:.4f}")
print(f"   (على المصدر الأصلي كانت: Accuracy={bundle['test_metrics']['accuracy']:.4f} "
      f"Recall={bundle['test_metrics']['recall']:.4f})")


# =========================================================
# 5. أثر وجود البروتوكول في الرابط
# =========================================================
# بيانات التدريب لا تحتوي ولا رابطاً واحداً يبدأ ببروتوكول، بينما كل
# روابط PhishTank تبدأ بـ http أو https. هذا اختلاف توزيع حقيقي،
# ونقيس أثره هنا بحذف البروتوكول وإعادة الفحص.
import re
stripped = [re.sub(r"^https?://", "", u) for u in phish_urls]
_, pred_s = predict(stripped)
print(f"\n🔬 أثر البروتوكول في نص الرابط:")
print(f"   كشف الروابط كما وردت (مع http/https) = {pred_p.mean():.4f}")
print(f"   كشفها بعد حذف البروتوكول             = {pred_s.mean():.4f}")
print(f"   الفرق = {abs(pred_s.mean() - pred_p.mean())*100:.2f} نقطة مئوية")


# =========================================================
# 6. تحليل ما فات النموذج: ما الذي يجمع بين الروابط الفائتة؟
# =========================================================
F = pd.DataFrame([extract_features(u) for u in phish_urls])   # خصائص روابط التصيد
missed_mask = pred_p == 0                                     # الروابط التي فاتت
bare = F["path_depth"].values == 0                            # روابط بلا مسار

print(f"\n🔍 تشريح الأخطاء:")
print(f"   روابط بلا مسار في PhishTank = {bare.mean()*100:.1f}% "
      f"| الكشف عليها = {pred_p[bare].mean():.4f}")
print(f"   روابط ذات مسار              = {(~bare).mean()*100:.1f}% "
      f"| الكشف عليها = {pred_p[~bare].mean():.4f}")
print(f"   متوسط الكلمات المشبوهة: فائت={F['suspicious_keyword_count'][missed_mask].mean():.2f} "
      f"مقابل مكشوف={F['suspicious_keyword_count'][~missed_mask].mean():.2f}")
print(f"   متوسط طول الرابط:       فائت={F['url_length'][missed_mask].mean():.1f} "
      f"مقابل مكشوف={F['url_length'][~missed_mask].mean():.1f}")
print("   ➜ الفائت روابط قصيرة خالية من أي إشارة معجمية، وغالباً مواقع")
print("     شرعية مخترقة تستضيف صفحة تصيد. لا يمكن لتحليل نص الرابط")
print("     وحده كشفها مهما تحسّن النموذج — وهذا حد نظري لا عيب تنفيذ.")


# =========================================================
# 7. أمثلة على ما فات النموذج
# =========================================================
missed = [(u, p) for u, p, k in zip(phish_urls, proba_p, pred_p) if k == 0]
missed.sort(key=lambda t: t[1])                             # الأبعد عن الكشف أولاً
print(f"\n❌ أمثلة على روابط تصيد فاتت النموذج ({len(missed)} إجمالاً):")
for u, p in missed[:8]:
    print(f"   احتمال={p:.3f}  {u[:78]}")

print("\n" + "=" * 66)
