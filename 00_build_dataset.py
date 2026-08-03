# =========================================================
# المرحلة 0 — إعادة بناء ملف البيانات بالمستخرج الرسمي
# =========================================================
# لماذا هذا الملف موجود؟
# الملف القديم Final_Phishing_Dataset_96k.csv بُني بدالة استخراج مختلفة
# عن feature_extractor.py، وثبت بالمقارنة أن 100% من الصفوف تختلف قيمها.
# لو دُرِّب النموذج على القديم واستُخدم المستخرج الجديد في الواجهة
# لأعطى نتائج عشوائية. لذلك نعيد بناء الملف من الروابط الخام مرة واحدة،
# ويصبح feature_extractor.py هو المصدر الوحيد للخصائص في المشروع كله.
# =========================================================

import sys                          # للتحكم في إعدادات بايثون
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # لعرض العربية في الشاشة بلا أخطاء

import os                           # للتعامل مع الملفات والمسارات
import time                         # لقياس الوقت المستغرق
import pandas as pd                 # مكتبة الجداول

# نستورد المستخرج الرسمي — ولا نكتب أي دالة استخراج هنا إطلاقاً
from feature_extractor import FEATURE_NAMES, extract_features, get_domain

SOURCE_FILE = "Final_Phishing_Dataset_96k.csv"   # الملف القديم (نأخذ منه عمودَي url و label فقط)
OUTPUT_FILE = "dataset_v2.csv"                   # الملف الجديد الذي سيُعتمد في التدريب
CHUNK_REPORT = 10000                             # كل كم رابط نطبع رسالة اطمئنان

# --- إعدادات البيانات التكميلية (Tranco) ---------------------------------
TRANCO_FILE = "data/top-1m.csv"   # قائمة تارانكو لأشهر المواقع (مصدر بحثي معتمد)
TRANCO_TOP_N = 60000              # من أي عمق في القائمة نسحب (كلما زاد قلّت شهرة الموقع)
TRANCO_SAMPLE = 5000              # كم نطاقاً مجرداً نضيف فعلياً


def load_raw_urls(path):
    """يقرأ الروابط الخام وتصنيفاتها فقط، ويتجاهل كل الخصائص القديمة."""
    if not os.path.exists(path):                               # نتأكد أن الملف المصدر موجود
        raise FileNotFoundError(f"لم يتم العثور على الملف: {path}")

    df = pd.read_csv(path, usecols=["url", "label"], low_memory=False)  # نقرأ عمودين فقط لتوفير الذاكرة
    print(f"✅ قُرئ {len(df)} رابطاً من {path}")

    df = df.dropna(subset=["url", "label"])                    # حذف أي صف ناقص
    df["url"] = df["url"].astype(str).str.strip()              # تنظيف المسافات الزائدة

    # المصدر الأصلي يحيط بعض الروابط بعلامات اقتباس مفردة أو مزدوجة.
    # لا نتركها لأن الرابط المخزَّن يجب أن يطابق ما يكتبه المستخدم فعلاً.
    before_q = df["url"].str.match(r"^['\"]").sum()             # كم رابطاً مقتبساً
    df["url"] = df["url"].str.strip("'").str.strip('"').str.strip()
    print(f"🧹 نُظّف {before_q} رابطاً كان محاطاً بعلامات اقتباس")

    # الرابط الحقيقي لا يحتوي مسافة خام إطلاقاً (المسافة تُرمَّز %20).
    # الصفوف التي فيها مسافة مشوّهة في المصدر، والأهم أن الـAPI يرفضها
    # وقت التشغيل، فتدريب النموذج عليها يعني تدريباً على مدخلات لا يمكن
    # أن تصل إليه أصلاً. اكتشف هذا التناقضَ اختبارُ التكامل (المرحلة 6).
    has_space = df["url"].str.contains(r"\s", regex=True)
    print(f"🚫 حُذف {has_space.sum()} رابطاً يحتوي مسافة خام (مشوّه في المصدر)")
    df = df[~has_space]

    df = df[df["url"] != ""]                                   # حذف الروابط الفارغة

    before = len(df)                                           # عدد الصفوف قبل حذف التكرار
    df = df.drop_duplicates(subset=["url"])                    # الرابط المكرر يسبب تسريباً بين التدريب والاختبار
    print(f"🔁 حُذف {before - len(df)} رابطاً مكرراً")

    df["label"] = df["label"].astype(float).round().astype(int)  # توحيد التصنيف كعدد صحيح 0 أو 1
    bad = set(df["label"].unique()) - {0, 1}                    # أي قيمة غير 0 و 1 خطأ في البيانات
    if bad:
        raise ValueError(f"قيم تصنيف غير متوقعة: {bad}")

    return df.reset_index(drop=True)                           # إعادة ترقيم الصفوف من الصفر


def add_bare_domains(df):
    """
    يضيف نطاقات شرعية مجرّدة (بلا مسار) من قائمة تارانكو.

    لماذا؟
    في البيانات الأصلية 0.35% فقط من الروابط الشرعية بلا مسار، مقابل
    8.60% من روابط التصيد. النتيجة أن النموذج تعلّم أن غياب المسار
    مؤشر تصيد، فصنّف google.com كتصيد باحتمال 0.98.
    هذه فجوة في مصدر البيانات لا عطل في الكود، وعلاجها إضافة أمثلة
    شرعية من هذا الشكل تحديداً.

    المصدر: Tranco (tranco-list.eu) — قائمة بحثية مبنية على متوسط
    ترتيب المواقع عبر عدة مصادر، مصمَّمة لمقاومة التلاعب.
    """
    if not os.path.exists(TRANCO_FILE):                        # القائمة اختيارية: لو غابت نكمل بدونها
        print(f"⚠️ لم تُوجد {TRANCO_FILE} — سيُبنى الملف بدون نطاقات مجردة.")
        return df

    tr = pd.read_csv(TRANCO_FILE, names=["rank", "domain"], nrows=TRANCO_TOP_N)
    print(f"\n📥 قُرئت أشهر {len(tr)} نطاقاً من قائمة تارانكو")

    # نستبعد أي نطاق ظهر في بيانات التصيد، حتى لا نُسمّي نطاقاً خبيثاً شرعياً
    phish_domains = set(df.loc[df["label"] == 1, "url"].apply(get_domain))
    clean = tr[~tr["domain"].isin(phish_domains)]              # الاستبعاد الفعلي
    print(f"   🚫 استُبعد {len(tr) - len(clean)} نطاقاً موجوداً أصلاً في بيانات التصيد")

    # نستبعد أيضاً ما هو موجود حرفياً في الملف الأصلي حتى لا يتكرر الرابط
    existing = set(df["url"])
    clean = clean[~clean["domain"].isin(existing)]

    # نأخذ عينة ثابتة (random_state) ليتكرر نفس الملف في كل تشغيل
    picked = clean.sample(min(TRANCO_SAMPLE, len(clean)), random_state=42)

    # نصف النطاقات نضيفها مجرّدة (google.com) والنصف الآخر بصيغة www
    # (www.google.com)، لأن المستخدم قد يلصق أي الشكلين، وتدريب النموذج
    # على شكل واحد فقط يجعله يرفض الشكل الآخر.
    domains = list(picked["domain"].values)
    urls = [d if i % 2 == 0 else f"www.{d}" for i, d in enumerate(domains)]
    print(f"   ➕ أُضيف {len(urls)} نطاقاً شرعياً بلا مسار (نصفها بصيغة www)")

    extra = pd.DataFrame({"url": urls, "label": 0})            # بناء الصفوف الجديدة
    out = pd.concat([df, extra], ignore_index=True)             # دمجها مع البيانات الأصلية
    return out.sample(frac=1, random_state=42).reset_index(drop=True)   # خلط الترتيب


def build_features(df):
    """يمرّ على كل رابط ويستخرج خصائصه الـ37 عبر المستخرج الرسمي."""
    rows = []                                                  # قائمة سنجمع فيها قاموس كل رابط
    start = time.time()                                        # بداية حساب الوقت

    print("⚙️ جاري استخراج الخصائص (دقيقتان تقريباً)...")
    for i, url in enumerate(df["url"], start=1):               # المرور على الروابط واحداً واحداً
        rows.append(extract_features(url))                     # استدعاء المستخرج الرسمي — لا شيء غيره
        if i % CHUNK_REPORT == 0:                              # كل عشرة آلاف رابط
            print(f"   🔄 {i} / {len(df)}")                     # رسالة تطمئن أن الكود يعمل

    out = pd.DataFrame(rows, columns=FEATURE_NAMES)            # تحويل القائمة إلى جدول بترتيب الخصائص الثابت
    out.insert(0, "url", df["url"].values)                     # نضع الرابط أول عمود (للتتبع فقط، لا يُدرَّب عليه)
    # عمود النطاق مطلوب في المرحلة 2 للتقسيم حسب النطاق (GroupShuffleSplit)
    out.insert(1, "domain", [get_domain(u) for u in df["url"]])
    out["label"] = df["label"].values                          # نضع التصنيف آخر عمود

    print(f"⏱️ استغرقت العملية {(time.time() - start) / 60:.2f} دقيقة")
    return out


def report(out):
    """يطبع ملخصاً للتحقق من سلامة الملف الجديد قبل اعتماده."""
    print("\n📋 فحص الملف الجديد:")
    print(f"   الأبعاد: {out.shape[0]} صف × {out.shape[1]} عمود")   # 37 خاصية + url + domain + label
    print(f"   خلايا فارغة: {int(out.isna().sum().sum())}")         # يجب أن تكون صفراً
    print(f"   نطاقات فريدة: {out['domain'].nunique()}")            # يُستخدم في التقسيم حسب النطاق

    counts = out["label"].value_counts().sort_index()               # توزيع الفئتين
    for lbl, cnt in counts.items():
        name = "تصيد" if lbl == 1 else "شرعي"
        print(f"   label={lbl} ({name}): {cnt} = {cnt / len(out) * 100:.2f}%")

    feats = out.drop(columns=["url", "domain", "label"])            # الخصائص وحدها
    dead = [c for c in feats.columns if feats[c].nunique() <= 1]     # خصائص ثابتة القيمة = ميتة
    print(f"   خصائص ميتة: {dead if dead else 'لا يوجد'}")

    # نسبة الروابط بلا مسار في كل فئة — الرقم الذي كنا نعالجه
    for lbl, name in [(0, "شرعي"), (1, "تصيد")]:
        s = out[out["label"] == lbl]
        print(f"   {name}: روابط بلا مسار = {(s['path_depth'] == 0).mean() * 100:.2f}%")


def main():
    print("=" * 60)
    print("المرحلة 0 — إعادة بناء البيانات بالمستخرج الرسمي")
    print("=" * 60)

    df = load_raw_urls(SOURCE_FILE)      # الخطوة 1: قراءة الروابط الخام
    df = add_bare_domains(df)            # الخطوة 2: إضافة نطاقات شرعية مجردة لسد فجوة البيانات
    out = build_features(df)             # الخطوة 3: استخراج الخصائص
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")  # الخطوة 4: الحفظ
    print(f"\n💾 حُفظ الملف الجديد: {OUTPUT_FILE}")
    report(out)                          # الخطوة 5: فحص النتيجة

    print("\n➜ اعتمد من الآن dataset_v2.csv فقط، والملف القديم مرجع تاريخي لا يُدرَّب عليه.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:               # أي خطأ يُطبع بوضوح بدل انهيار البرنامج
        print(f"❌ {e}")
