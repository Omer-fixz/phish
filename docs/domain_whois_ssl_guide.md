# دليل فحص النطاق وWHOIS وSSL/TLS
# Domain Age / WHOIS / SSL — How They Work

---

## هل كان النظام يفحصهم قبل؟

**لا.** النظام كان يعمل بنمط **Lexical Only** — تحليل نص الرابط فقط بدون أي اتصال بالإنترنت.  
بعد هذا التحديث أصبح يفحص **3 طبقات إضافية** عند كل طلب فحص.

---

## 1. فحص عمر النطاق — Domain Age (WHOIS)

### كيف يعمل؟

```
الرابط المُدخل
      ↓
استخراج hostname (مثال: secure-login.tk)
      ↓
استعلام WHOIS عبر مكتبة python-whois
      ↓
قراءة حقل creation_date من سجل WHOIS
      ↓
domain_age_days = اليوم الحالي - تاريخ التسجيل
```

### لماذا هذا مؤشر مهم؟

| الحالة | المعنى |
|---|---|
| `domain_age_days > 365` | نطاق قديم ← غالباً شرعي |
| `domain_age_days < 30` | نطاق جديد جداً ← مؤشر تصيد |
| `domain_age_days = -1` | تعذّر الحصول على المعلومة (WHOIS محجوب أو النطاق غير مسجّل) |

مواقع التصيد تُنشأ وتُستخدم خلال أيام ثم تُهجر. المواقع الشرعية مسجّلة منذ سنوات.

### مكان الكود

```python
# src/feature_extractor.py
def _get_domain_age_days(hostname: str) -> int:
    ...
f["domain_age_days"] = _get_domain_age_days(hostname)
```

### مثال

```
google.com         → domain_age_days ≈ 9800  (منذ 1997)
secure-paypal.tk   → domain_age_days ≈ 3     (أُنشئ قبل 3 أيام)
```

---

## 2. فحص شهادة SSL/TLS

### كيف يعمل؟

```
hostname
      ↓
اتصال TCP على المنفذ 443
      ↓
TLS Handshake عبر ssl.create_default_context()
      ↓
قراءة الشهادة: cert = conn.getpeercert()
      ↓
استخراج 4 خصائص من الشهادة
```

### الخصائص المستخرجة

| الخاصية | القيمة | المعنى |
|---|---|---|
| `ssl_valid` | 1 / 0 | الشهادة موجودة وصالحة الآن |
| `ssl_days_left` | عدد الأيام / -1 | كم يوم متبقي قبل انتهاء الشهادة |
| `ssl_self_signed` | 1 / 0 | الشهادة موقّعة ذاتياً (Self-Signed) بدون جهة تصديق |
| `ssl_wildcard` | 1 / 0 | شهادة بدل `*.domain.com` — أحياناً مؤشر مشبوه |

### لماذا هذا مؤشر مهم؟

- **ssl_valid = 0**: الموقع لا يملك شهادة موثوقة — مؤشر قوي للتصيد أو الموقع مزيف.
- **ssl_days_left قصير جداً**: مواقع التصيد لا تجدد شهاداتها لأنها مؤقتة.
- **ssl_self_signed = 1**: أي شخص يستطيع إنشاء شهادة ذاتية — لا تعني أي ثقة.
- **ملاحظة**: امتلاك HTTPS لا يعني الأمان — كثير من مواقع التصيد الحديثة تستخدم HTTPS.

### مكان الكود

```python
# src/feature_extractor.py
def _get_ssl_info(hostname: str) -> dict:
    ...
ssl_info = _get_ssl_info(hostname)
f["ssl_valid"]       = ssl_info["ssl_valid"]
f["ssl_days_left"]   = ssl_info["ssl_days_left"]
f["ssl_self_signed"] = ssl_info["ssl_self_signed"]
f["ssl_wildcard"]    = ssl_info["ssl_wildcard"]
```

### مثال

```
github.com          → ssl_valid=1, ssl_days_left=180, ssl_self_signed=0
evil-login.tk       → ssl_valid=0, ssl_days_left=-1,  ssl_self_signed=0
local-test-server   → ssl_valid=0, ssl_days_left=-1,  ssl_self_signed=1
```

---

## 3. ملخص الخصائص الجديدة في FEATURE_NAMES

```python
# src/feature_extractor.py — المجموعة 8 (أضيفت بعد التحديث)
"domain_age_days",   # عمر النطاق بالأيام (WHOIS)
"ssl_valid",         # هل الشهادة صالحة؟
"ssl_days_left",     # أيام متبقية للشهادة
"ssl_self_signed",   # هل هي ذاتية التوقيع؟
"ssl_wildcard",      # هل هي شهادة بدل؟
```

---

## 4. تدفق الفحص الكامل بعد التحديث

```
رابط مُدخل من المستخدم
        ↓
┌─────────────────────────────────────────────────┐
│           feature_extractor.py                  │
│                                                 │
│  المجموعة 1-7: تحليل نص الرابط (Offline)       │
│  ← url_length, entropy, brand_in_wrong_place…  │
│                                                 │
│  المجموعة 8: فحص الشبكة (Online)               │
│  ← WHOIS → domain_age_days                     │
│  ← SSL/TLS → ssl_valid, ssl_days_left...       │
└─────────────────────────────────────────────────┘
        ↓
متجه من 43 خاصية → النموذج (XGBoost) → النتيجة
```

---

## 5. تنبيهات مهمة

> **⚠️ الأداء:** فحص WHOIS و SSL يضيف ~2-5 ثوانٍ لكل طلب بسبب الاتصال بالشبكة.  
> الخصائص الأصلية (1-7) تبقى أقل من 1ms.

> **⚠️ إعادة التدريب مطلوبة:** النموذج الحالي `model.joblib` دُرِّب على 38 خاصية فقط.  
> بعد إضافة الخصائص الجديدة يجب إعادة تشغيل `scripts/04_train_model.py`.

> **⚠️ القيم الافتراضية:** إن تعذّر الاتصال (موقع غير موجود، WHOIS محجوب)  
> تُعطى القيم: `domain_age_days=-1`, `ssl_valid=0`, `ssl_days_left=-1`.

---

## 6. تثبيت المكتبات الجديدة

```bash
pip install python-whois cryptography
```

أو من ملف المتطلبات:

```bash
pip install -r requirements.txt
```

الملفات المعدَّلة:
- `src/feature_extractor.py` — الدوال والخصائص الجديدة
- `requirements.txt` — إضافة `python-whois` و`cryptography`
