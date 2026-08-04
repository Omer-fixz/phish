# التوثيق الشامل للخوارزميات والمكتبات والأدوات المستخدمة (Algorithms, Libraries & Tools Reference)

هذا المستند يوفر حشداً وتفصيلاً شاملاً لجميع **الخوارزميات**، **المكتبات**، و**الأدوات البرمجية** المستخدمة في هذا المشروع، مع تحديد **أماكن وجودها الدقيقة في الكود**، والسبب المنهجي لاستخدام كل منها.

---

## 🤖 1. الخوارزميات (Algorithms & ML Models)

تمت المقارنة بين 4 خوارزميات تعلم آلة متباينة في [`scripts/04_train_model.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py) وتحديد الخوارزمية الفائزة:

| # | الخوارزمية | مكان وجودها في الكود | الوظيفة ودواعي الاستخدام |
|---|---|---|---|
| 1 | **XGBoost (Extreme Gradient Boosting)** | [`scripts/04_train_model.py#L38`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py#L38) | **الخوارزمية الأساسية الفائزة والنسخة النهائية للنموذج.** تتميز بالسرعة الفائقة والدقة العالية في التصنيف ودعم تقنيات تعزيز الأشجار (Gradient Boosting). |
| 2 | **Random Forest Classifier** | [`scripts/04_train_model.py#L37`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py#L37) | خوارزمية تجميعية (Ensemble) قائمة على دمج عدة أشجار قرار، تم استخدامها في المقارنة المنهجية. |
| 3 | **Decision Tree Classifier** | [`scripts/04_train_model.py#L36`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py#L36) | خوارزمية شجرة القرار المفردة، استُخدمت كنموذج مرجعي أساسي (Baseline Model). |
| 4 | **Logistic Regression** | [`scripts/04_train_model.py#L35`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py#L35) | نموذج تصنيف خطي، تم تضمينه للمقارنة وضبط التطبيع القياسي (StandardScaler). |
| 5 | **GroupShuffleSplit** | [`scripts/04_train_model.py#L33`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py#L33) | خوارزمية تقسيم البيانات العادل بناءً على اسم النطاق لمنع تسريب البيانات (`Data Leakage`) بين التدريب والاختبار. |
| 6 | **TreeSHAP / Feature Contribution** | [`src/api/main.py#L260`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/api/main.py#L260) | خوارزمية حساب أسباب اتخاذ القرار وتوضيح أعلى 3 خصائص مؤثرة في النتيجة (Model Explainability). |
| 7 | **Shannon Entropy Algorithm** | [`src/feature_extractor.py#L180`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/feature_extractor.py#L180) | خوارزمية قياس العشوائية الرياضية (إنتروبيا شانون) في نص الرابط والنطاق لكشف الأسماء المولدة آلياً. |

---

## 📚 2. المكتبات وحزم البرمجة (Libraries & Dependencies)

جميع المكتبات موثقة ومحددة في ملفات التثبيت [`requirements.txt`](file:///c:/Users/Omer%20fixz/Desktop/phish/requirements.txt) و [`requirements-api.txt`](file:///c:/Users/Omer%20fixz/Desktop/phish/requirements-api.txt):

### أ. مكتبات الذكاء الاصطناعي ومعالجة البيانات:
- **`scikit-learn`:** 
  - *مكان الاستخدام:* [`scripts/04_train_model.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py)
  - *الوظيفة:* تقسيم المجموعات، حساب المقاييس (`F1-Score`, `ROC-AUC`, `Recall`, `Precision`).
- **`xgboost`:** 
  - *مكان الاستخدام:* [`scripts/04_train_model.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py)
  - *الوظيفة:* تدريب وتوليد نموذج الـ XGBoost المعالج.
- **`pandas`:** 
  - *مكان الاستخدام:* [`scripts/01_build_dataset.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/01_build_dataset.py) و [`scripts/03_eda.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/03_eda.py)
  - *الوظيفة:* قراءة، تنظيف، معالجة، وإدارة الجداول والمصفوفات.
- **`numpy`:** 
  - *مكان الاستخدام:* [`src/feature_extractor.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/feature_extractor.py) و [`src/api/main.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/api/main.py)
  - *الوظيفة:* التحويل السريع للمصفوفات الرقمية وعمليات المتجهات.
- **`joblib`:** 
  - *مكان الاستخدام:* [`scripts/04_train_model.py#L23`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/04_train_model.py#L23) و [`src/api/main.py#L25`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/api/main.py#L25)
  - *الوظيفة:* تسلسل وحفظ وقراءة حزمة النموذج المدرب بكفاءة وسرعة.

### ب. مكتبات السيرفر والـ API والواجهة:
- **`fastapi`:** 
  - *مكان الاستخدام:* [`src/api/main.py#L27`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/api/main.py#L27)
  - *الوظيفة:* الإطار الرئيسي لإتاحة الـ RESTful API واستقبال الطلبات وتوثيق Swagger (`/docs`).
- **`uvicorn`:** 
  - *مكان الاستخدام:* [`run_api.py#L8`](file:///c:/Users/Omer%20fixz/Desktop/phish/run_api.py#L8)
  - *الوظيفة:* خادم الـ ASGI عالي الأداء لتشغيل تطبيق FastAPI.
- **`pydantic`:** 
  - *مكان الاستخدام:* [`src/api/main.py#L30`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/api/main.py#L30)
  - *الوظيفة:* التحقق القياسي من هيكل وجودة المدخلات والمخرجات للـ API.

### ج. مكتبات التحليل البصري والاختبارات:
- **`matplotlib` & `seaborn`:** 
  - *مكان الاستخدام:* [`scripts/03_eda.py`](file:///c:/Users/Omer%20fixz/Desktop/phish/scripts/03_eda.py)
  - *الوظيفة:* توليد صور الرسوم البيانية والجداول الإحصائية المخرجة بدقة عالية.
- **`pytest`:** 
  - *مكان الاستخدام:* المجلد [`tests/`](file:///c:/Users/Omer%20fixz/Desktop/phish/tests/)
  - *الوظيفة:* إطار تشغيل الاختبارات الآلية والتحقق من جودة واستقرار الكود.

---

## 🛠️ 3. الأدوات والتقنيات (Tools & Stack)

1. **لغة البرمجة (Python 3.10+):**
   - اللغة المعيارية المستخدمة في النظام بالكامل.

2. **لغات وتطوير الـ Web (HTML5 / CSS3 / JavaScript):**
   - *مكان الاستخدام:* [`src/api/static/index.html`](file:///c:/Users/Omer%20fixz/Desktop/phish/src/api/static/index.html)
   - صياغة وتصميم واجهة المستخدم العربية المتفاعلية بدون أي اعتمادات خارجية ثقيلة (Vanilla Stack).

3. **أداة Vercel CLI:**
   - *مكان الاستخدام:* ملفات التكوييد [`vercel.json`](file:///c:/Users/Omer%20fixz/Desktop/phish/vercel.json) و [`.vercelignore`](file:///c:/Users/Omer%20fixz/Desktop/phish/.vercelignore)
   - رفع وتوزيع الموقع مجاناً على السحابة (Cloud Deployment).

4. **نظام إدارة النسخ (Git & GitHub):**
   - تتبع وتوثيق التغييرات ورفع المشروع على مستودع GitHub.
