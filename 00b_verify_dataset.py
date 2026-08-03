# =========================================================
# فحص التطابق بين ملف البيانات والمستخرج الرسمي
# =========================================================
# هذا السكربت يجيب على سؤال واحد فقط:
# «هل قيم الخصائص المحفوظة في dataset_v2.csv هي نفسها التي ينتجها
#  feature_extractor.py اليوم؟»
# إن كان الجواب لا، فالنموذج سيُدرَّب على أرقام مختلفة عن التي ستصله
# من الواجهة، وهو العطل الذي أفسد الملف القديم. شغّله بعد أي تعديل
# على feature_extractor.py وقبل التدريب.
# =========================================================

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # لعرض العربية بلا أخطاء

import pandas as pd                                          # مكتبة الجداول
from feature_extractor import FEATURE_NAMES, extract_features, get_domain

DATA_FILE = "dataset_v2.csv"   # الملف المراد فحصه
SAMPLE_SIZE = 2000             # عدد الروابط التي نفحصها (عينة تكفي لكشف أي اختلاف)
TOLERANCE = 1e-6               # فرق مسموح به بسبب تقريب الأرقام العشرية فقط


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)            # قراءة الملف
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)  # عينة عشوائية ثابتة ليتكرر الاختبار بنفس النتيجة
    print(f"🔍 فحص {len(sample)} رابطاً من {DATA_FILE}")

    fresh = pd.DataFrame([extract_features(u) for u in sample["url"]],
                         columns=FEATURE_NAMES)              # إعادة استخراج الخصائص من الصفر
    stored = sample[FEATURE_NAMES].reset_index(drop=True).astype(float)  # الخصائص كما هي محفوظة
    fresh = fresh.astype(float)                              # توحيد النوع قبل المقارنة

    problems = []                                            # قائمة الخصائص غير المتطابقة
    for col in FEATURE_NAMES:                                # المرور على كل خاصية
        mismatch = ((fresh[col] - stored[col]).abs() > TOLERANCE).sum()  # عدد الصفوف المختلفة
        if mismatch:
            problems.append((col, mismatch))                 # نسجّل الخاصية وعدد اختلافاتها

    dom_mismatch = sum(get_domain(u) != d for u, d in
                       zip(sample["url"], sample["domain"]))  # فحص عمود النطاق أيضاً

    print("\n" + "=" * 50)
    if not problems and dom_mismatch == 0:                   # الحالة السليمة
        print("✅ تطابق تام: الملف والمستخرج ينتجان نفس القيم.")
        print("   يمكن الانتقال إلى مرحلة التدريب بأمان.")
        sys.exit(0)

    print("❌ يوجد اختلاف — لا تُدرِّب النموذج قبل إصلاحه:")   # الحالة الخطرة
    for col, n in problems:                                  # طباعة كل خاصية مختلفة
        print(f"   - {col}: {n} صف مختلف من {len(sample)}")
    if dom_mismatch:
        print(f"   - domain: {dom_mismatch} صف مختلف")
    print("   ➜ الحل: إعادة تشغيل 00_build_dataset.py لبناء الملف من جديد.")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print(f"❌ لم يتم العثور على {DATA_FILE}. شغّل 00_build_dataset.py أولاً.")
        sys.exit(1)
