# مخطط هيكلة المشروع (Schema)

> **حالة هذا المستند: خطة مقترحة لم تُنفَّذ بعد.**
> الهيكل الحالي مسطّح: 20 ملفاً في مجلد واحد. هذا المستند يصف الهيكل
> المقترح، وما ينتقل إلى أين، وما يجب تعديله في الكود، وكيف نتحقق أن
> شيئاً لم ينكسر. لا يُنفَّذ شيء منه قبل موافقتك.

---

## 1. المشكلة في الهيكل الحالي

```
phish/
├── 00_build_dataset.py      ← سكربت
├── 00b_verify_dataset.py    ← سكربت
├── 01_eda.py                ← سكربت
├── train_model.py           ← سكربت (بلا رقم، فترتيبه غير واضح)
├── feature_extractor.py     ← مكتبة أساسية، مختلطة بالسكربتات
├── main.py                  ← خدمة، اسمها لا يدل عليها
├── index.html               ← واجهة
├── test_*.py × 4            ← اختبارات
├── dataset_v2.csv           ← بيانات مُولَّدة
├── Final_Phishing_...csv    ← بيانات خام
├── model.joblib             ← مخرَج تدريب
├── feature_names.json       ← مخرَج تدريب
├── figures/                 ← مخرجات
├── data/                    ← بيانات خارجية
└── *.md × 3                 ← توثيق
```

أربع مشكلات عملية:

1. **لا يُعرف من أين يبدأ القارئ.** `train_model.py` بلا رقم بينما ما قبله مرقّم، و`00b` ترقيم مرتجل.
2. **الكود المشترك مختلط بالسكربتات.** `feature_extractor.py` هو قلب المشروع لكنه يبدو كملف عادي بين عشرين ملفاً.
3. **البيانات والمخرجات والمصدر في مستوى واحد.** الخام والمُولَّد والنموذج والرسومات متجاورة بلا فصل.
4. **المسارات نسبية للمجلد الحالي.** كل سكربت يكتب `"dataset_v2.csv"` مباشرة، فيفشل إن شُغّل من مجلد آخر.

---

## 2. الهيكل المقترح

```
phish/
│
├── README.md                    # نقطة الدخول: النتائج وخطوات التشغيل
├── STRUCTURE.md                 # هذا الملف
├── requirements.txt             # المكتبات
├── .gitignore
├── run_api.py                   # تشغيل الخادم بأمر واحد
│
├── src/                         # ← الكود المشترك (يُستورَد ولا يُشغَّل مباشرة)
│   ├── __init__.py
│   ├── paths.py                 # كل مسارات المشروع في مكان واحد
│   ├── feature_extractor.py     # المصدر الوحيد للخصائص الـ37
│   └── api/
│       ├── __init__.py
│       ├── main.py              # تطبيق FastAPI
│       └── static/
│           └── index.html       # واجهة المستخدم
│
├── scripts/                     # ← خط المعالجة، يُشغَّل بالترتيب الرقمي
│   ├── 01_build_dataset.py
│   ├── 02_verify_dataset.py
│   ├── 03_eda.py
│   └── 04_train_model.py
│
├── tests/                       # ← كل الاختبارات
│   ├── test_feature_extractor.py
│   ├── test_api.py
│   ├── test_integration.py
│   └── test_external.py
│
├── data/                        # ← بيانات (مستثناة من git بالكامل)
│   ├── raw/                     # ما يُنزَّل أو يُستلَم كما هو
│   │   ├── Final_Phishing_Dataset_96k.csv
│   │   ├── top-1m.csv           # تارانكو
│   │   ├── phishtank.csv.gz
│   │   └── openphish.txt
│   └── processed/               # ما يُولِّده المشروع
│       └── dataset_v2.csv
│
├── models/                      # ← مخرجات التدريب (مستثناة من git)
│   ├── model.joblib
│   └── feature_names.json
│
├── reports/                     # ← مخرجات للرسالة (مستثناة من git)
│   ├── figures/                 # الرسومات PNG بدقة 300
│   └── tables/                  # الجداول CSV
│
├── docs/                        # ← توثيق
│   ├── قاموس_البيانات.md
│   └── البرومبت_المصحح.md
│
└── archive/                     # ← كود قديم للتوثيق فقط، لا يُستخدم
    └── sorting_OLD_DO_NOT_USE.py
```

### لماذا هذا التقسيم تحديداً

| المجلد | القاعدة التي يجيب عنها | السؤال الذي يختصره |
|-|-|-|
| `src/` | كود يُستورَد | «أين منطق المشروع؟» |
| `scripts/` | كود يُشغَّل بالترتيب | «من أين أبدأ؟» — الرقم يجيب |
| `tests/` | كود يتحقق | «كيف أتأكد أنه يعمل؟» |
| `data/raw` | مدخلات لا تُعدَّل أبداً | «ما الذي جاء من الخارج؟» |
| `data/processed` | مخرجات قابلة لإعادة التوليد | «ما الذي أستطيع حذفه وإعادة بنائه؟» |
| `models/` | نموذج + ترتيب خصائصه معاً | «ما الذي أنشره؟» |
| `reports/` | ما يُنسخ إلى الرسالة | «أين الرسومات والجداول؟» |
| `docs/` | نصوص للقراءة | «أين الشرح؟» |
| `archive/` | لا يُستخدم | «لماذا هذا الملف موجود؟» |

الفصل بين `raw` و`processed` هو الأهم: **يمكنك حذف `processed` و`models` و`reports` كاملة وإعادة توليدها بأمرين**، بينما `raw` لا يُعاد توليده أبداً.

---

## 3. جدول الترحيل الكامل

| # | من | إلى | الأداة |
|-|-|-|-|
| 1 | `feature_extractor.py` | `src/feature_extractor.py` | `git mv` |
| 2 | `main.py` | `src/api/main.py` | `git mv` |
| 3 | `index.html` | `src/api/static/index.html` | `git mv` |
| 4 | `00_build_dataset.py` | `scripts/01_build_dataset.py` | `git mv` |
| 5 | `00b_verify_dataset.py` | `scripts/02_verify_dataset.py` | `git mv` |
| 6 | `01_eda.py` | `scripts/03_eda.py` | `git mv` |
| 7 | `train_model.py` | `scripts/04_train_model.py` | `git mv` |
| 8 | `test_feature_extractor.py` | `tests/test_feature_extractor.py` | `git mv` |
| 9 | `test_api.py` | `tests/test_api.py` | `git mv` |
| 10 | `test_integration.py` | `tests/test_integration.py` | `git mv` |
| 11 | `test_external.py` | `tests/test_external.py` | `git mv` |
| 12 | `قاموس_البيانات.md` | `docs/قاموس_البيانات.md` | `git mv` |
| 13 | `البرومبت_المصحح.md` | `docs/البرومبت_المصحح.md` | `git mv` |
| 14 | `Final_Phishing_Dataset_96k.csv` | `data/raw/` | `mv` (غير متتبَّع) |
| 15 | `data/top-1m.csv` | `data/raw/top-1m.csv` | `mv` |
| 16 | `data/phishtank.csv.gz` | `data/raw/phishtank.csv.gz` | `mv` |
| 17 | `data/openphish.txt` | `data/raw/openphish.txt` | `mv` |
| 18 | `data/tranco_top1m.csv.zip` | `data/raw/tranco_top1m.csv.zip` | `mv` |
| 19 | `dataset_v2.csv` | `data/processed/dataset_v2.csv` | `mv` |
| 20 | `model.joblib` | `models/model.joblib` | `mv` |
| 21 | `feature_names.json` | `models/feature_names.json` | `mv` |
| 22 | `figures/*.png` | `reports/figures/` | `mv` |
| 23 | `figures/*.csv` | `reports/tables/` | `mv` |
| — | `archive/sorting_OLD_DO_NOT_USE.py` | يبقى مكانه | — |

`git mv` يحفظ تاريخ الملف، فلا يظهر النقل كحذف وإنشاء.

---

## 4. ملفات جديدة تُنشأ

### `src/paths.py` — كل المسارات في مكان واحد

هذا الملف هو ما يحل مشكلة «السكربت يفشل إن شُغّل من مجلد آخر».
يحسب جذر المشروع من موقعه هو، فتصبح كل المسارات مطلقة:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # جذر المشروع

RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
FIGURES_DIR   = ROOT / "reports" / "figures"
TABLES_DIR    = ROOT / "reports" / "tables"

SOURCE_CSV    = RAW_DIR / "Final_Phishing_Dataset_96k.csv"
TRANCO_CSV    = RAW_DIR / "top-1m.csv"
PHISHTANK_GZ  = RAW_DIR / "phishtank.csv.gz"
OPENPHISH_TXT = RAW_DIR / "openphish.txt"
DATASET_CSV   = PROCESSED_DIR / "dataset_v2.csv"
MODEL_FILE    = MODELS_DIR / "model.joblib"
FEATURES_JSON = MODELS_DIR / "feature_names.json"
INDEX_HTML    = ROOT / "src" / "api" / "static" / "index.html"

def ensure_dirs():
    """ينشئ المجلدات الناقصة قبل الكتابة فيها."""
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

### `run_api.py` — تشغيل الخادم بأمر واحد

```python
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### `src/__init__.py` و `src/api/__init__.py`
ملفان فارغان يجعلان `src` حزمة قابلة للاستيراد.

---

## 5. التعديلات المطلوبة في الكود

### أ. رأس كل سكربت واختبار (سطران)

الملفات داخل `scripts/` و`tests/` تحتاج معرفة جذر المشروع لتستورد `src`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
```

**لماذا هذا الحل ولا نستخدم تثبيت الحزمة (`pip install -e .`)؟**
لأنه يضيف `pyproject.toml` وخطوة تثبيت إضافية على من يشغّل المشروع.
سطران في رأس الملف أبسط لمشروع بهذا الحجم، ولا يحتاج المستخدم فعل شيء.

### ب. تغيير الاستيرادات

| الحالي | الجديد |
|-|-|
| `from feature_extractor import ...` | `from src.feature_extractor import ...` |
| `import main` (في الاختبارات) | `from src.api import main` |

### ج. استبدال المسارات النصية بثوابت `paths.py`

| الملف | الحالي | الجديد |
|-|-|-|
| `01_build_dataset.py` | `"Final_Phishing_Dataset_96k.csv"` | `paths.SOURCE_CSV` |
| | `"dataset_v2.csv"` | `paths.DATASET_CSV` |
| | `"data/top-1m.csv"` | `paths.TRANCO_CSV` |
| `02_verify_dataset.py` | `"dataset_v2.csv"` | `paths.DATASET_CSV` |
| `03_eda.py` | `"dataset_v2.csv"` | `paths.DATASET_CSV` |
| | `"figures"` | `paths.FIGURES_DIR` |
| | `figures/descriptive_stats.csv` | `paths.TABLES_DIR` |
| `04_train_model.py` | `"dataset_v2.csv"` | `paths.DATASET_CSV` |
| | `"model.joblib"` | `paths.MODEL_FILE` |
| | `"feature_names.json"` | `paths.FEATURES_JSON` |
| | `"figures"` | `paths.FIGURES_DIR` / `paths.TABLES_DIR` |
| `src/api/main.py` | `"model.joblib"` | `paths.MODEL_FILE` |
| | `"feature_names.json"` | `paths.FEATURES_JSON` |
| | `"index.html"` | `paths.INDEX_HTML` |
| `tests/test_integration.py` | `"dataset_v2.csv"` | `paths.DATASET_CSV` |
| | `"model.joblib"` | `paths.MODEL_FILE` |
| `tests/test_external.py` | `"data/phishtank.csv.gz"` | `paths.PHISHTANK_GZ` |
| | `"data/openphish.txt"` | `paths.OPENPHISH_TXT` |
| | `"data/top-1m.csv"` | `paths.TRANCO_CSV` |

**فائدة جانبية**: بعد هذا التعديل يعمل أي سكربت من أي مجلد:
`python scripts/03_eda.py` أو `cd scripts && python 03_eda.py` كلاهما يعمل.

### د. تعديل خاص في `03_eda.py` و`04_train_model.py`

حالياً تُحفظ الجداول CSV داخل `figures/` مع الرسومات. بعد الترحيل
تنفصل: الرسومات في `reports/figures/` والجداول في `reports/tables/`.

---

## 6. `.gitignore` المخطط

```gitignore
# ============ البيانات ============
# لا يُرفع أي شيء داخل data/: الخام كبير (31 ميجابايت)
# والمُعالَج يُعاد توليده بأمر واحد
data/

# ============ مخرجات التدريب ============
models/
reports/

# ============ بايثون ============
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/
.pytest_cache/

# ============ أسرار ============
.env
*.key
*.pem

# ============ نظام التشغيل والمحررات ============
.vscode/
.idea/
.DS_Store
Thumbs.db
desktop.ini
```

**تغيير مهم عن الحالي**: بدل استثناء ملفات بأسمائها (`dataset_v2.csv`،
`model.joblib`…) نستثني **مجلدات كاملة**. هذا أمتن: أي ملف جديد يُولَّد
داخلها لاحقاً يُستثنى تلقائياً بلا تعديل.

يُحفظ `.gitkeep` فارغ داخل `data/raw` و`data/processed` و`models`
و`reports/figures` و`reports/tables` مع استثناء مضاد، حتى يعرف من
يستنسخ المشروع أين توضع الملفات. (سطر `!**/.gitkeep`.)

---

## 7. `README.md` المخطط — العناوين

1. **العنوان ووصف سطر واحد**
2. **النتائج** — جدول المقاييس + نتيجة الاختبار الخارجي
3. **الهيكل** — شجرة مختصرة مع سطر لكل مجلد + إحالة إلى `STRUCTURE.md`
4. **التثبيت** — `pip install -r requirements.txt`
5. **البيانات** — أوامر تنزيل تارانكو وPhishTank وOpenPhish إلى `data/raw/`
6. **التشغيل** — السكربتات الأربعة بالترتيب ثم `python run_api.py`
7. **الاختبارات** — الأوامر الأربعة وعدد اختبارات كل ملف
8. **القرارات المنهجية** — الخمسة (التقسيم حسب النطاق، مستخرج واحد، العتبة، الخصائص الميتة، لا اتصال بالشبكة)
9. **حدود النظام** — الستة، بينها التنبيه أن نسبة الإنذارات الكاذبة المعتمدة 11.5% لا 0.13%
10. **الترخيص والإسناد** — مصادر البيانات الثلاثة

---

## 8. خطة التحقق بعد التنفيذ

تُنفَّذ بالترتيب، وأي فشل يوقف العملية:

| # | الأمر | المتوقع |
|-|-|-|
| 1 | `python tests/test_feature_extractor.py` | 51 / 51 |
| 2 | `python scripts/02_verify_dataset.py` | تطابق تام |
| 3 | `python tests/test_api.py` | 46 / 46 |
| 4 | `python tests/test_integration.py` | 13 / 13 |
| 5 | `python run_api.py` ثم فتح `/` | الصفحة تعمل والفحص يعطي نتيجة |
| 6 | `python scripts/03_eda.py` | 5 رسومات في `reports/figures/` |
| 7 | `git status` | لا ملفات بيانات أو نماذج غير متتبَّعة ظاهرة |

**اختبار إضافي حاسم**: تشغيل سكربت من مجلد آخر للتأكد أن المسارات
المطلقة تعمل:
```bash
cd /tmp && python "C:/Users/Omer fixz/Desktop/phish/scripts/02_verify_dataset.py"
```

**لا يُعاد التدريب.** النموذج الحالي ينتقل كما هو، ويثبت اختبار التكامل
أنه ما زال يعطي نفس النتائج بعد النقل. لو أُعيد التدريب لتغيّرت الأرقام
في الرسالة بلا سبب.

---

## 9. المخاطر

| الخطر | الاحتمال | التخفيف |
|-|-|-|
| كسر استيراد لم يُلاحَظ | متوسط | خطة التحقق أعلاه تشغّل كل ملف فعلياً |
| مسار نصي منسي في مكان ما | متوسط | البحث عن `"data/` و`.csv"` و`.joblib"` في كل الملفات بعد التعديل |
| ضياع تاريخ git للملفات | منخفض | `git mv` لا `mv` للملفات المتتبَّعة |
| اختلاف نتائج النموذج بعد النقل | منخفض جداً | اختبار التكامل يقارن 1000 رابط على المسارين |

**نقطة تراجع**: التنفيذ كله في إيداع واحد. للتراجع:
`git reset --hard HEAD~1` ثم إعادة ملفات البيانات يدوياً (غير متتبَّعة).

---

## 10. ما ينتقل ولا يتغير محتواه

هذه الملفات تُنقل فقط، ولا يُعدَّل سطر واحد من منطقها:

- `feature_extractor.py` — لا يقرأ أي ملف، فلا مسارات فيه
- `archive/sorting_OLD_DO_NOT_USE.py` — مؤرشف
- `docs/*.md` — توثيق
- `index.html` — مساراته نسبية للخادم (`/predict`) لا للقرص

أي تعديل عليها في هذه العملية يكون خطأً.
