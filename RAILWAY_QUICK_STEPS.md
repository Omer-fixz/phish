# 🚀 خطوات النشر السريع على Railway

## ✅ **الإعداد جاهز!**

جميع ملفات النشر موجودة ومرفوعة على GitHub:
- ✅ `requirements.txt`
- ✅ `Procfile`
- ✅ `runtime.txt`
- ✅ `.railwayignore`
- ✅ `models/model.joblib`

---

## 🎯 **خطوات النشر (5 دقائق)**

### 1️⃣ **افتح Railway**
```
https://railway.app
```

### 2️⃣ **سجل دخول**
- اضغط **"Login"**
- اختر **"Login with GitHub"**

### 3️⃣ **أنشئ مشروع جديد**
- اضغط **"New Project"**
- اختر **"Deploy from GitHub repo"**
- ابحث عن: `Omer-fixz/phish`
- اضغط **"Deploy Now"**

### 4️⃣ **انتظر البناء** (2-3 دقائق)
تابع في **Deployments** → **View Logs**

يجب أن ترى:
```
✅ حُمّل النموذج: XGBoost ...
✅ عتبة القرار المعتمدة = 0.38
INFO: Application startup complete.
```

### 5️⃣ **احصل على الرابط**
- اذهب إلى **Settings** → **Networking**
- اضغط **"Generate Domain"**
- انسخ الرابط: `https://phish-production-xxxx.up.railway.app`

---

## ✅ **اختبار التطبيق**

### 1. افتح الرابط:
```
https://your-app.railway.app/
```

### 2. اختبر `/health`:
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

### 3. جرب فحص رابط:
- الصق: `https://www.google.com`
- النتيجة: ✅ **أخضر** (آمن)

---

## 🎉 **تم النشر!**

التطبيق الآن يعمل على الإنترنت ويمكن لأي شخص الوصول إليه!

---

## 📝 **ملاحظات:**

- 🆓 **الخطة المجانية**: 500 ساعة/شهر
- ⏱️ قد ينام بعد 10 دقائق بدون استخدام
- 🔄 **تحديثات تلقائية**: كل push على GitHub → نشر جديد
- 📊 **مراقبة**: Metrics & Logs في لوحة Railway

---

## 🆘 **مشاكل؟**

راجع الدليل الكامل: `RAILWAY_DEPLOY.md`

---

**جاهز للنشر! 🚀**
