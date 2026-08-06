# 🚀 نشر المشروع على Railway - الدليل الكامل

## ✅ **الخطوة 1: التحضير (تم!)**

جميع الملفات المطلوبة موجودة وجاهزة:
- ✅ `requirements.txt` - المكتبات المطلوبة مع الإصدارات
- ✅ `Procfile` - أمر تشغيل السيرفر
- ✅ `runtime.txt` - إصدار Python
- ✅ `.railwayignore` - ملفات لا نحتاجها في Production
- ✅ `railway.json` - إعدادات Railway
- ✅ `models/model.joblib` - النموذج المُدرَّب

---

## 📋 **الخطوة 2: إنشاء حساب Railway**

1. اذهب إلى: https://railway.app
2. اضغط **"Start a New Project"**
3. سجل دخول عبر GitHub (موصى به)

---

## 🔗 **الخطوة 3: ربط المشروع من GitHub**

### الطريقة 1: عبر واجهة Railway (الأسهل)

1. في Railway، اضغط **"New Project"**
2. اختر **"Deploy from GitHub repo"**
3. اختر repository: `Omer-fixz/phish`
4. اضغط **"Deploy Now"**

### الطريقة 2: عبر Git (يدوي)

إذا كنت تريد النشر مباشرة:
```bash
# 1. تثبيت Railway CLI
npm i -g @railway/cli

# 2. تسجيل الدخول
railway login

# 3. ربط المشروع
railway link

# 4. النشر
railway up
```

---

## ⚙️ **الخطوة 4: إعدادات Railway**

بعد إنشاء المشروع، ستحتاج إلى:

### 1. **التحقق من Build Settings:**
   - **Build Command**: (تلقائي) `pip install -r requirements.txt`
   - **Start Command**: (من Procfile) `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

### 2. **إضافة متغيرات البيئة (اختياري):**
   في قسم **Variables**:
   ```
   PORT=8000  # تلقائي من Railway
   ENABLE_TRAINING=0  # تعطيل صفحة التدريب في Production
   ```

### 3. **إعدادات إضافية:**
   - **Region**: اختر أقرب منطقة (مثل: Europe/Frankfurt)
   - **Health Check Path**: `/health`

---

## 🔍 **الخطوة 5: مراقبة النشر**

### ما الذي سيحدث:
1. ✅ Railway يسحب الكود من GitHub
2. ✅ يثبت المكتبات من `requirements.txt`
3. ✅ يحمّل النموذج `models/model.joblib`
4. ✅ يشغل السيرفر على المنفذ التلقائي

### في لوحة Railway:
- تابع **Deployments** → **Logs** لمشاهدة التقدم
- انتظر حتى ترى:
  ```
  ✅ حُمّل النموذج: XGBoost (depth=6, lr=0.1) بـ 37 خاصية
  ✅ عتبة القرار المعتمدة = 0.38
  INFO: Application startup complete.
  ```

---

## 🌐 **الخطوة 6: الحصول على الرابط**

بعد نجاح النشر:

1. في Railway، اضغط **"Settings"**
2. في قسم **"Networking"**:
   - اضغط **"Generate Domain"**
   - أو اختر **"Custom Domain"** إن كان لديك نطاق

3. ستحصل على رابط مثل:
   ```
   https://phish-production-xxxx.up.railway.app
   ```

4. **اختبر الرابط:**
   ```
   https://your-app.railway.app/health
   ```
   يجب أن يرجع:
   ```json
   {
     "status": "ok",
     "model": "XGBoost",
     "features": 37,
     "threshold": 0.38
   }
   ```

---

## ✅ **الخطوة 7: اختبار التطبيق**

### 1. **افتح الواجهة:**
```
https://your-app.railway.app/
```

### 2. **جرب فحص رابط:**
- الصق رابط: `https://www.google.com`
- اضغط **"فحص"**
- النتيجة: **✅ رابط آمن** (أخضر)

### 3. **جرب رابط مشبوه:**
- الصق: `http://paypal-verify-account.tk/login`
- النتيجة: **⛔ رابط خطر** (أحمر)

---

## 🔄 **التحديثات التلقائية**

Railway يتابع GitHub تلقائيًا:

1. كل **push** إلى `main` → نشر تلقائي جديد
2. لتعطيل النشر التلقائي:
   - Settings → Deployments → **Disable Auto Deploy**

---

## 📊 **مراقبة الأداء**

في لوحة Railway:

### 1. **Metrics:**
   - استخدام CPU
   - استخدام الذاكرة (RAM)
   - عدد الطلبات

### 2. **Logs:**
   - رسائل النظام
   - أخطاء Python
   - طلبات الفحص

### 3. **Usage:**
   - عدد ساعات التشغيل
   - Bandwidth المستخدم

---

## ⚠️ **ملاحظات مهمة**

### 1. **الخطة المجانية:**
   - ✅ **500 ساعة شهريًا** مجانًا (حوالي 20 يوم)
   - ✅ **مناسبة للعرض التوضيحي**
   - ⚠️ قد ينام التطبيق بعد 10 دقائق بدون استخدام

### 2. **حجم الملفات:**
   - النموذج: ~1.24 MB ✅
   - المكتبات: ~250 MB ✅
   - إجمالي: أقل من 500 MB ✅

### 3. **الأداء:**
   - الفحص بدون شبكة: ~1-2 ms ⚡
   - الفحص مع شبكة: ~5-10 ثانية (WHOIS + SSL)
   - الذاكرة المستخدمة: ~200-300 MB

### 4. **الأمان:**
   - ✅ HTTPS تلقائي من Railway
   - ✅ لا توجد متغيرات بيئة حساسة
   - ✅ النموذج محلي (لا استدعاءات API خارجية)

---

## 🐛 **حل المشاكل الشائعة**

### 1. **"Application failed to respond"**
   - **السبب**: النموذج ثقيل أو المكتبات تستغرق وقتًا
   - **الحل**: انتظر 2-3 دقائق للتحميل الأول

### 2. **"Module not found"**
   - **السبب**: مكتبة ناقصة في `requirements.txt`
   - **الحل**: تحقق من Logs وأضف المكتبة

### 3. **"Port already in use"**
   - **السبب**: Railway يحدد `$PORT` تلقائيًا
   - **الحل**: تأكد من استخدام `$PORT` في Procfile

### 4. **"Model file not found"**
   - **السبب**: `models/model.joblib` غير موجود في Git
   - **الحل**: تأكد من رفع النموذج على GitHub

---

## 🎯 **الخطوات التالية (اختياري)**

### 1. **إضافة نطاق مخصص:**
```
Settings → Networking → Custom Domain
```

### 2. **إضافة SSL Certificate:**
(تلقائي من Railway)

### 3. **إضافة CI/CD:**
يمكنك إضافة GitHub Actions للاختبار قبل النشر

### 4. **مراقبة متقدمة:**
ربط مع Sentry أو LogRocket

---

## 📝 **الأوامر المفيدة (Railway CLI)**

```bash
# عرض logs مباشرة
railway logs

# فتح Dashboard
railway open

# ربط مشروع محلي
railway link

# نشر يدوي
railway up

# عرض المتغيرات
railway variables

# حالة المشروع
railway status
```

---

## 🆘 **الدعم**

إذا واجهت مشاكل:

1. **Railway Docs**: https://docs.railway.app
2. **Railway Discord**: https://discord.gg/railway
3. **GitHub Issues**: في repository المشروع

---

## ✅ **Checklist النشر النهائي**

قبل النشر، تأكد:

- [ ] ✅ جميع الملفات في GitHub (خاصة `models/model.joblib`)
- [ ] ✅ `requirements.txt` محدث بالإصدارات
- [ ] ✅ `Procfile` يستخدم `$PORT`
- [ ] ✅ `.railwayignore` يستثني الملفات غير الضرورية
- [ ] ✅ النموذج يعمل محليًا بنجاح
- [ ] ✅ حجم النموذج أقل من 5 MB
- [ ] ✅ اختبار `/health` endpoint

---

**جاهز للنشر! 🚀**

ابدأ الآن من الخطوة 2 في هذا الدليل.
