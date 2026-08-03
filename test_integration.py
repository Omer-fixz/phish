# =========================================================
# المرحلة 6 — اختبار التكامل
# =========================================================
# السؤال الذي يجيب عنه هذا الملف سؤال واحد فقط:
#
#   «هل الرابط الواحد يعطي النتيجة ذاتها في مسار التدريب وفي الـAPI؟»
#
# لماذا هذا الاختبار تحديداً؟
# أشهر عطل في مشاريع كشف التصيد ليس ضعف النموذج، بل اختلاف صامت
# بين طريقة استخراج الخصائص وقت التدريب وطريقتها وقت التشغيل.
# النتيجة: نموذج دقته 90% في الرسالة، وسلوكه شبه عشوائي في الواجهة،
# بلا أي رسالة خطأ تدل على السبب.
#
# وقد حدث هذا فعلاً في هذا المشروع: الملف الأول بُني بدالة استخراج
# مختلفة عن feature_extractor.py، وكانت 100% من الصفوف تختلف قيمها.
# هذا الاختبار هو ما يمنع تكرار ذلك.
#
# المسارَان اللذان نقارنهما:
#   المسار (أ) — وقت التدريب: القيم المحفوظة في dataset_v2.csv
#                تُمرَّر على النموذج مباشرة، كما يفعل train_model.py.
#   المسار (ب) — وقت التشغيل: الرابط النصي يُرسل إلى /predict،
#                فيستخرج الخادم خصائصه بنفسه ثم يتنبأ.
# لو اختلف المساران في أي رقم، فالنظام معطوب مهما بدت نتائجه جيدة.
# =========================================================

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json                                # لقراءة ترتيب الخصائص
import joblib                              # لتحميل النموذج
import numpy as np                         # للعمليات الرقمية
import pandas as pd                        # للجداول
from sklearn.model_selection import GroupShuffleSplit   # لإعادة بناء نفس التقسيم
from fastapi.testclient import TestClient  # لاستدعاء الـAPI بلا تشغيل خادم منفصل

import main                                # تطبيق الـAPI نفسه
from feature_extractor import extract_features

# ---------------------------------------------------------
# إعدادات
# ---------------------------------------------------------
DATA_FILE = "dataset_v2.csv"   # نفس الملف الذي دُرِّب عليه النموذج
SAMPLE_SIZE = 1000             # عدد الروابط التي نمررها على المسارين
# الـAPI يُرجع الاحتمال مقرَّباً إلى أربع خانات عشرية، فأدق فرق يمكن
# ملاحظته بين المسارين هو نصف خانة أي 0.00005. نضع السماحية 0.0001
# لأن أي فرق أصغر ناتج عن التقريب وحده لا عن اختلاف حقيقي في الحساب.
# أي فرق أكبر من ذلك يعني اختلافاً فعلياً بين التدريب والتشغيل.
TOLERANCE = 1e-4
SEED = 42                      # نفس بذرة التقسيم في train_model.py

client = TestClient(main.app)
passed = failed = 0


def check(label, ok, detail=""):
    """يسجّل نتيجة اختبار واحد."""
    global passed, failed
    if ok:
        passed += 1
        print(f"  [ نجح ] {label}")
    else:
        failed += 1
        print(f"  [ فشل ] {label}  {detail}")


def api_predict(url):
    """
    يستدعي الـAPI لرابط واحد ويُرجع رده.

    ملاحظة: نصفّر عدّاد تحديد معدل الطلبات قبل كل استدعاء، لأن الحد
    وُضع لحماية الخادم من الإساءة في الإنتاج، وليس قيداً على صحة
    النتائج. تعطيله هنا لا يؤثر على ما نقيسه إطلاقاً.
    """
    main._hits.clear()
    r = client.post("/predict", json={"url": url})
    if r.status_code != 200:
        return None
    return r.json()


# =========================================================
# 1. إعادة بناء نفس التقسيم المستخدم في التدريب
# =========================================================
print("=" * 60)
print("المرحلة 6 — اختبار التكامل بين التدريب والتشغيل")
print("=" * 60)

bundle = joblib.load("model.joblib")                      # النموذج المحفوظ
names = json.load(open("feature_names.json", encoding="utf-8"))  # ترتيب الخصائص
df = pd.read_csv(DATA_FILE, low_memory=False)             # البيانات

# نعيد نفس خطوات التقسيم بنفس البذرة، فنحصل على نفس مجموعة الاختبار
X, y, g = df[names], df["label"], df["domain"]
_, temp = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(X, y, g))
_, test_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED)
                   .split(X.iloc[temp], y.iloc[temp], g.iloc[temp]))
test_df = df.iloc[temp[test_rel]]                          # مجموعة الاختبار كاملة

sample = test_df.sample(min(SAMPLE_SIZE, len(test_df)), random_state=SEED)
print(f"\n📦 عينة الاختبار: {len(sample)} رابطاً من مجموعة الاختبار")
print(f"   النموذج: {bundle['model_name']} | العتبة: {bundle['threshold']} "
      f"| الخصائص: {len(names)}")


# =========================================================
# 2. اتفاق الإعدادات بين الطرفين
# =========================================================
print("\n=== 1. اتفاق الإعدادات ===")
health = client.get("/health").json()
check("الـAPI يستخدم نفس عدد الخصائص", health["features"] == len(names),
      f"API={health['features']} النموذج={len(names)}")
check("الـAPI يستخدم عتبة النموذج المحفوظة",
      abs(health["threshold"] - bundle["threshold"]) < 1e-9,
      f"API={health['threshold']} النموذج={bundle['threshold']}")
check("ترتيب الخصائص في الـAPI مطابق للنموذج", main.NAMES == bundle["feature_names"])


# =========================================================
# 3. المقارنة الكبرى: المسار (أ) مقابل المسار (ب)
# =========================================================
print(f"\n=== 2. مقارنة {len(sample)} رابطاً على المسارين ===")

# --- المسار (أ): القيم المحفوظة في الملف تُمرَّر على النموذج مباشرة ---
proba_train = bundle["model"].predict_proba(sample[names].values)[:, 1]

# --- المسار (ب): الرابط النصي يمر عبر الـAPI كاملاً ---
proba_api, feature_mismatches, errors = [], [], []
for i, (_, row) in enumerate(sample.iterrows(), start=1):
    res = api_predict(row["url"])
    if res is None:                                        # الـAPI رفض الرابط
        errors.append(row["url"])
        proba_api.append(np.nan)
        continue
    proba_api.append(res["phishing_probability"])

    # فحص إضافي: قيم الخصائص المعروضة في الرد يجب أن تطابق ما في الملف
    for f in res["top_features"]:
        stored = float(row[f["name"]])
        if abs(stored - f["value"]) > 1e-3:
            feature_mismatches.append((row["url"], f["name"], stored, f["value"]))

    if i % 250 == 0:
        print(f"   🔄 {i} / {len(sample)}")

proba_api = np.array(proba_api)

check(f"كل الروابط قُبلت في الـAPI ({len(errors)} مرفوضة)", len(errors) == 0,
      f"أمثلة: {errors[:3]}")

# الفرق في الاحتمال بين المسارين
valid = ~np.isnan(proba_api)                               # نتجاهل المرفوضة إن وُجدت
diff = np.abs(proba_train[valid].astype(float) - proba_api[valid])
max_diff = diff.max() if len(diff) else 0
n_diff = int((diff > TOLERANCE).sum())

check(f"احتمالات المسارين متطابقة (أكبر فرق = {max_diff:.8f})", n_diff == 0,
      f"{n_diff} رابطاً مختلفاً")

# القرار النهائي (تصيد/شرعي) لا بد أن يتطابق أيضاً
thr = bundle["threshold"]
label_train = (proba_train[valid] >= thr).astype(int)
label_api = (proba_api[valid] >= thr).astype(int)
n_label_diff = int((label_train != label_api).sum())
check(f"القرارات النهائية متطابقة ({len(label_train)} رابطاً)", n_label_diff == 0,
      f"{n_label_diff} قراراً مختلفاً")

check(f"قيم الخصائص في ردود الـAPI تطابق الملف "
      f"({len(feature_mismatches)} اختلاف)", len(feature_mismatches) == 0,
      f"أمثلة: {feature_mismatches[:2]}")


# =========================================================
# 4. هل يُعيد النظام المنشور نفس الدقة المعلنة في الرسالة؟
# =========================================================
print("\n=== 3. إعادة إنتاج الدقة المعلنة عبر الـAPI ===")
y_true = sample["label"].values[valid]
acc_api = (label_api == y_true).mean()                     # الدقة كما يقيسها النظام الحي
acc_train = (label_train == y_true).mean()                 # الدقة عبر مسار التدريب
reported = bundle["test_metrics"]["accuracy"]              # الدقة المحفوظة وقت التدريب

print(f"   الدقة عبر مسار التدريب  = {acc_train:.4f}")
print(f"   الدقة عبر الـAPI الحي   = {acc_api:.4f}")
print(f"   الدقة المعلنة في الرسالة = {reported:.4f} (على مجموعة الاختبار كاملة)")

check("الدقة عبر الـAPI تساوي الدقة عبر مسار التدريب", abs(acc_api - acc_train) < 1e-9)
# العينة جزء من مجموعة الاختبار، فيُتوقع فرق بسيط عن الرقم المعلن لا أكثر من 5%
check("الدقة عبر الـAPI قريبة من الرقم المعلن", abs(acc_api - reported) < 0.05,
      f"الفرق = {abs(acc_api - reported):.4f}")


# =========================================================
# 5. الثبات: نفس الرابط ينتج نفس النتيجة دائماً
# =========================================================
print("\n=== 4. ثبات النتائج ===")
u = "paypal.secure-login.tk/verify/account.php?id=99"
r1, r2 = api_predict(u), api_predict(u)
check("استدعاءان متتاليان يعطيان نفس الاحتمال",
      r1["phishing_probability"] == r2["phishing_probability"])
check("والتفسير نفسه أيضاً", r1["top_features"] == r2["top_features"])

# الاستخراج المباشر يجب أن يعطي نفس متجه الخصائص الذي استعمله الـAPI
direct = extract_features(u)
vec = np.array([[direct[n] for n in names]], dtype=float)
proba_direct = float(main.MODEL.predict_proba(vec)[0][1])
check("الاستخراج المباشر يعطي نفس احتمال الـAPI",
      abs(round(proba_direct, 4) - r1["phishing_probability"]) < 1e-9,
      f"مباشر={proba_direct:.6f} API={r1['phishing_probability']}")


# =========================================================
# 6. روابط خارج البيانات كلياً (سلوك واقعي)
# =========================================================
print("\n=== 5. روابط لم يرها النموذج إطلاقاً ===")
external = {
    "www.uofk.edu/faculties/science": 0,
    "github.com/python/cpython/issues": 0,
    "www.bbc.co.uk/news/world-middle-east": 0,
    "microsoft.com": 0,
    "secure-paypal-verify.tk/login/account/update.php": 1,
    "apple.id-verify.xyz/signin/confirm": 1,
    "http://192.168.1.55/wp-admin/bank/login.php": 1,
    "xn--pypal-4ve.com/webscr?cmd=_login-submit": 1,
}
correct = 0
for url, expected in external.items():
    res = api_predict(url)
    got = res["label"]
    ok = got == expected
    correct += ok
    mark = "✔" if ok else "✘"
    print(f"   {mark} {res['prediction']:11s} ({res['phishing_probability']:.3f})  {url[:50]}")
check(f"أصاب {correct} من {len(external)} رابطاً خارجياً", correct == len(external))


# =========================================================
# النتيجة
# =========================================================
print("\n" + "=" * 60)
print(f"النتيجة النهائية: نجح {passed} | فشل {failed}")
if failed == 0:
    print("✅ مسار التدريب ومسار التشغيل متطابقان تماماً.")
    print("   النموذج الذي قِيس في الرسالة هو نفسه الذي يعمل في الواجهة.")
else:
    print("❌ يوجد اختلاف بين التدريب والتشغيل — لا تعتمد نتائج الرسالة قبل إصلاحه.")
print("=" * 60)
sys.exit(1 if failed else 0)
