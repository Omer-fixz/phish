# =========================================================
# المرحلة 3 — واجهة برمجية (API) بـ FastAPI
# =========================================================
# ماذا يفعل هذا الملف؟
#   يستقبل رابطاً عبر HTTP، يستخرج خصائصه بالمستخرج الرسمي نفسه
#   المستخدم في التدريب، يمرّرها على النموذج المحفوظ، ويُرجع:
#   التصنيف + نسبة الثقة + أهم ثلاث خصائص أدّت للقرار مع قيمها.
#
# قاعدة أمنية غير قابلة للتفاوض:
#   النظام لا يفتح الرابط ولا يطلب محتواه ولا يتبع إعادة التوجيه.
#   كل شيء يُحسب من نص الرابط فقط. لو فتحنا الرابط لصار الخادم نفسه
#   ضحية للصفحة الخبيثة، ولأصبح أداة لمهاجمة مواقع أخرى.
# =========================================================

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # لعرض العربية في السجلات

import hashlib                                  # لتوليد بصمة قصيرة للرابط بدل تسجيله كاملاً
import json                                      # لقراءة ملف أسماء الخصائص
import logging                                   # لتسجيل الأحداث
import os                                        # للتعامل مع الملفات
import time                                      # لقياس نوافذ تحديد معدل الطلبات
from collections import defaultdict, deque       # هياكل بيانات عدّاد الطلبات
import numpy as np                               # للمصفوفات الرقمية
import joblib                                    # لتحميل النموذج المحفوظ

from fastapi import FastAPI, HTTPException, Request  # إطار العمل ورفع أخطاء HTTP
from fastapi.middleware.cors import CORSMiddleware   # ضبط المواقع المسموح لها بالاستدعاء
from fastapi.responses import JSONResponse, HTMLResponse  # للتحكم في شكل الرد وتقديم الصفحة
from pydantic import BaseModel, Field, field_validator  # للتحقق من صحة المدخلات

# المستخرج الرسمي — نفسه المستخدم في التدريب، وهذا شرط صحة النتائج
from src.paths import MODEL_FILE, FEATURES_JSON, INDEX_HTML
from src.feature_extractor import extract_features, FEATURE_NAMES

# ---------------------------------------------------------
# إعدادات
# ---------------------------------------------------------
# المسارات تأتي من src/paths.py فيعمل الخادم من أي مجلد
MODEL_PATH = MODEL_FILE               # ملف النموذج المحفوظ من المرحلة 2
FEATURES_FILE = FEATURES_JSON         # ترتيب الخصائص المعتمد
INDEX_FILE = INDEX_HTML               # صفحة الواجهة التي يقدّمها الخادم
DEFAULT_THRESHOLD = 0.5               # تُستخدم فقط لو كان النموذج قديماً بلا عتبة محفوظة
MAX_URL_LENGTH = 2000                 # أقصى طول مقبول للرابط (حماية من المدخلات الضخمة)
TOP_K = 3                             # عدد الخصائص المفسِّرة التي نُرجعها

# --- إعدادات التأمين (المرحلة 5) -----------------------------------------
RATE_LIMIT = 30                       # أقصى عدد طلبات فحص لكل عنوان IP
RATE_WINDOW = 60                      # خلال هذا العدد من الثواني (دقيقة واحدة)
MAX_TRACKED_IPS = 10000               # سقف عدد العناوين المتتبَّعة في الذاكرة
ALLOWED_ORIGINS = [                   # المواقع المسموح لها باستدعاء الـAPI
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("phishing-api")


# =========================================================
# 1. تحميل النموذج مرة واحدة عند بدء التشغيل
# =========================================================
# نحمّله هنا لا داخل الدالة، لأن تحميله مع كل طلب يجعل الاستجابة بطيئة جداً.
def load_bundle():
    """يحمّل النموذج وأسماء الخصائص ويتأكد من تطابقهما."""
    if not MODEL_PATH.exists():                              # النموذج غير موجود
        raise RuntimeError(f"{MODEL_PATH} غير موجود — شغّل scripts/04_train_model.py أولاً")

    bundle = joblib.load(MODEL_PATH)                         # تحميل الحزمة المحفوظة

    with open(FEATURES_FILE, encoding="utf-8") as fh:        # قراءة ترتيب الخصائص
        names = json.load(fh)

    # فحص حاسم: لو اختلف ترتيب الخصائص بين الملفين فالنتائج ستكون عشوائية
    if names != bundle["feature_names"]:
        raise RuntimeError("ترتيب الخصائص في feature_names.json لا يطابق النموذج")

    # فحص ثانٍ: كل خاصية يتوقعها النموذج يجب أن ينتجها المستخرج
    missing = [n for n in names if n not in FEATURE_NAMES]
    if missing:
        raise RuntimeError(f"خصائص يتوقعها النموذج ولا ينتجها المستخرج: {missing}")

    log.info(f"حُمّل النموذج: {bundle['model_name']} ({bundle['params']}) "
             f"بـ {len(names)} خاصية")
    return bundle, names


BUNDLE, NAMES = load_bundle()          # التحميل يحدث مرة واحدة عند تشغيل الخادم
MODEL = BUNDLE["model"]                # النموذج نفسه
SCALER = BUNDLE.get("scaler")          # المطبّع (None للنماذج الشجرية)

# العتبة تأتي من النموذج نفسه لا من رقم مكتوب هنا، لأنها اختيرت على
# مجموعة التحقق أثناء التدريب. هكذا لا يمكن أن تختلف عتبة التقييم
# عن عتبة التشغيل، وأي إعادة تدريب تحدّثها تلقائياً.
THRESHOLD = float(BUNDLE.get("threshold", DEFAULT_THRESHOLD))
log.info(f"عتبة القرار المعتمدة = {THRESHOLD}")


# =========================================================
# 2. شكل الطلب والرد (Pydantic)
# =========================================================
class URLRequest(BaseModel):
    """شكل الطلب القادم من الواجهة: حقل واحد اسمه url."""
    url: str = Field(..., min_length=3, max_length=MAX_URL_LENGTH,
                     description="الرابط المراد فحصه")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """تحقق مبدئي من صيغة الرابط قبل أن يصل إلى النموذج."""
        v = v.strip()                                        # إزالة المسافات الزائدة
        if not v:                                            # رابط فارغ
            raise ValueError("الرابط فارغ")
        if " " in v:                                         # الروابط الحقيقية لا تحوي مسافات
            raise ValueError("الرابط يحتوي على مسافة — تأكد من نسخه كاملاً")
        if "." not in v:                                     # لا بد من نقطة واحدة على الأقل
            raise ValueError("الرابط لا يحتوي على نقطة — ليس رابطاً صالحاً")
        # نمنع البروتوكولات الخطرة مثل javascript: و data: و file:
        low = v.lower()
        for bad in ("javascript:", "data:", "file:", "vbscript:"):
            if low.startswith(bad):
                raise ValueError(f"بروتوكول غير مسموح: {bad}")
        return v


class FeatureExplanation(BaseModel):
    """خاصية واحدة من الخصائص المفسِّرة للقرار."""
    name: str            # اسم الخاصية كما في القاموس
    value: float         # قيمتها في هذا الرابط تحديداً
    contribution: float  # مقدار دفعها للقرار (موجب = نحو التصيد، سالب = نحو الشرعية)
    direction: str       # "نحو التصيد" أو "نحو الشرعية"


class PredictionResponse(BaseModel):
    """شكل الرد الذي تستهلكه الواجهة."""
    url: str                                 # الرابط كما أُرسل (نص خام، لا يُفتح)
    prediction: str                          # "phishing" أو "legitimate"
    label: int                               # 1 تصيد / 0 شرعي
    confidence: float                        # نسبة الثقة في القرار المتخذ (0 إلى 1)
    phishing_probability: float              # احتمال أن يكون تصيداً تحديداً
    threshold: float                         # العتبة المستخدمة، للشفافية
    top_features: list[FeatureExplanation]   # أهم الخصائص التي أدّت للقرار


# =========================================================
# 3. تفسير القرار: من أين جاء الحكم؟
# =========================================================
def explain(vector: np.ndarray) -> list[dict]:
    """
    يحسب مساهمة كل خاصية في قرار هذا الرابط تحديداً.

    XGBoost يوفّر حساب قيم SHAP بدقة تامة عبر pred_contribs، وهي
    المساهمة الحقيقية لكل خاصية في هذا التنبؤ بالذات (لا الأهمية
    العامة للنموذج). إن كان النموذج من نوع آخر نرجع للأهمية العامة.
    """
    try:
        import xgboost as xgb                                 # مستورد هنا لأنه لازم لهذا الفرع فقط
        booster = MODEL.get_booster()                         # الوصول للنواة الداخلية
        dm = xgb.DMatrix(vector, feature_names=NAMES)         # تحويل المتجه لصيغة XGBoost
        contribs = booster.predict(dm, pred_contribs=True)[0] # آخر قيمة هي الانحياز (bias)
        values = contribs[:-1]                                # نحذف الانحياز ونُبقي الخصائص
    except Exception:                                         # نموذج ليس XGBoost أو خطأ ما
        if hasattr(MODEL, "feature_importances_"):            # النماذج الشجرية
            values = MODEL.feature_importances_
        else:                                                 # النماذج الخطية
            values = MODEL.coef_[0]

    # نرتّب حسب حجم التأثير بغض النظر عن اتجاهه، ونأخذ الأقوى
    order = np.argsort(np.abs(values))[::-1][:TOP_K]

    return [{
        "name": NAMES[i],
        "value": float(vector[0][i]),
        "contribution": round(float(values[i]), 4),
        "direction": "نحو التصيد" if values[i] > 0 else "نحو الشرعية",
    } for i in order]


# =========================================================
# 4. التطبيق ومساراته
# =========================================================
app = FastAPI(
    title="نظام كشف روابط التصيد الاحتيالي",
    description="فحص معجمي للروابط بلا أي اتصال بالموقع المفحوص",
    version="1.0.0",
)


# =========================================================
# المرحلة 5 (أ) — ضبط CORS
# =========================================================
# CORS يحدد أي المواقع يُسمح لها باستدعاء هذا الـAPI من متصفح المستخدم.
# لا نستخدم "*" إطلاقاً: لو سمحنا للجميع لأمكن لأي موقع خبيث أن يبني
# واجهة مزيفة تستهلك خادمنا وتنسب نتائجه لنفسه.
# الواجهة عندنا تُقدَّم من نفس الخادم أصلاً، فلا تحتاج CORS إطلاقاً،
# وهذه القائمة موجودة فقط لحالة تشغيل الواجهة على منفذ تطوير منفصل.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,     # قائمة محددة بالاسم لا "*"
    allow_credentials=False,           # لا نستخدم كوكيز ولا جلسات، فلا داعي لها
    allow_methods=["GET", "POST"],     # لا نسمح بـ DELETE أو PUT لأننا لا نستخدمها
    allow_headers=["Content-Type"],    # الترويسة الوحيدة التي تحتاجها الواجهة
)


# =========================================================
# المرحلة 5 (ب) — تحديد معدل الطلبات (Rate Limiting)
# =========================================================
# لماذا؟ بدون حد أقصى يستطيع شخص واحد إرسال آلاف الطلبات في الدقيقة
# فيستهلك المعالج ويوقف الخدمة عن بقية المستخدمين، أو يستعمل النظام
# كأداة لفحص قوائم روابط ضخمة مجاناً.
#
# الطريقة: نافذة زمنية منزلقة. نحتفظ لكل عنوان IP بأوقات طلباته
# خلال آخر دقيقة فقط، ونرفض الطلب إن تجاوز عددها الحد المسموح.
# هذا حل بسيط يعمل في الذاكرة ويكفي لمشروع بخادم واحد. النشر على
# عدة خوادم يحتاج مخزناً مشتركاً مثل Redis (يُذكر كتطوير مستقبلي).
_hits = defaultdict(deque)                 # قاموس: عنوان IP ← أوقات طلباته الأخيرة


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """يعترض كل طلب ويرفضه إن تجاوز صاحبه الحد المسموح."""
    # نطبّق الحد على مسار الفحص فقط، لا على الصفحة ولا على /health
    if request.url.path != "/predict":
        return await call_next(request)

    ip = request.client.host if request.client else "unknown"  # هوية صاحب الطلب
    now = time.time()                                          # الوقت الحالي بالثواني
    window = _hits[ip]                                         # سجل طلبات هذا العنوان

    while window and now - window[0] > RATE_WINDOW:            # حذف الطلبات الأقدم من النافذة
        window.popleft()

    if len(window) >= RATE_LIMIT:                              # تجاوز الحد
        retry = int(RATE_WINDOW - (now - window[0])) + 1        # بعد كم ثانية يُسمح له
        log.warning(f"تجاوز حد الطلبات: {ip} ({len(window)} طلباً)")
        return JSONResponse(
            status_code=429,                                   # 429 = طلبات كثيرة جداً
            content={"detail": f"طلبات كثيرة جداً. حاول بعد {retry} ثانية."},
            headers={"Retry-After": str(retry)},               # ترويسة قياسية يفهمها المتصفح
        )

    window.append(now)                                         # تسجيل هذا الطلب

    # تنظيف دوري: نمنع نمو القاموس بلا حدود لو تعرّض الخادم لعناوين كثيرة
    if len(_hits) > MAX_TRACKED_IPS:
        for key in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW]:
            del _hits[key]

    return await call_next(request)                            # السماح بمرور الطلب


# =========================================================
# المرحلة 5 (ج) — ترويسات الحماية
# =========================================================
@app.middleware("http")
async def security_headers(request: Request, call_next):
    """يضيف ترويسات تحدّ من أثر أي ثغرة محتملة في الواجهة."""
    response = await call_next(request)                        # ننفّذ الطلب أولاً

    # يمنع المتصفح من تخمين نوع الملف وتنفيذه كسكربت
    response.headers["X-Content-Type-Options"] = "nosniff"
    # يمنع وضع صفحتنا داخل إطار في موقع آخر (هجوم Clickjacking)
    response.headers["X-Frame-Options"] = "DENY"
    # يمنع تسريب الرابط المفحوص إلى مواقع أخرى عبر ترويسة Referer
    response.headers["Referrer-Policy"] = "no-referrer"
    # سياسة المحتوى: تحدد مصادر السكربتات المسموح بها فقط
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'"                               # تأكيد منع التأطير
    )
    return response


# موقع التدريب: صفحة مستقلة على /training مخصّصة للتدريب فقط.
# فُصلت عن صفحة الفحص لأن وظيفتهما مختلفة تماماً: الأولى للمستخدم
# النهائي، وهذه لمن يبني النموذج.
#
# لا تُسجَّل مساراتها إلا إذا كان ENABLE_TRAINING مضبوطاً، لأنها تستهلك
# المعالج وتكتب ملفات نماذج. وعدم تسجيلها أقوى من رفض الطلب: المسار
# غير موجود أصلاً، ولا يظهر في /docs، فلا يعلم أحد بوجود الميزة.
from src.api import training as training_module

if training_module.ENABLED:
    app.include_router(training_module.router)
    log.info("موقع التدريب مُفعَّل على /training")
else:
    log.info("موقع التدريب مُعطَّل (اضبط ENABLE_TRAINING=1 لتفعيله محلياً)")


@app.get("/", response_class=HTMLResponse)
def home():
    """
    يقدّم صفحة الواجهة (المرحلة 4).

    نقدّمها من نفس الخادم عمداً: لو فُتح index.html من القرص مباشرة
    (file://) لمنع المتصفح طلباته إلى الـAPI بسبب سياسة المصدر الواحد.
    تقديمها من هنا يجعل الصفحة والـAPI على نفس المصدر فلا حاجة لـ CORS.
    """
    if not INDEX_FILE.exists():                              # الواجهة غير موجودة
        raise HTTPException(status_code=404, detail="index.html غير موجود")
    with open(INDEX_FILE, encoding="utf-8") as fh:           # قراءة الصفحة
        return HTMLResponse(fh.read())


@app.get("/health")
def health():
    """مسار بسيط للتأكد أن الخادم يعمل والنموذج محمّل."""
    return {
        "status": "ok",
        "model": BUNDLE["model_name"],
        "features": len(NAMES),
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: URLRequest):
    """
    المسار الرئيسي: يفحص رابطاً واحداً ويُرجع القرار مع تفسيره.

    ملاحظة أمنية: لا يوجد في هذه الدالة أي استدعاء شبكي.
    الرابط يُعامل كنص فقط ولا يُفتح إطلاقاً.
    """
    try:
        feats = extract_features(req.url)                     # استخراج الخصائص (نفس دالة التدريب)
    except Exception as e:                                    # رابط مشوّه جداً
        log.warning(f"فشل استخراج الخصائص: {type(e).__name__}")
        raise HTTPException(status_code=400,
                            detail="تعذّر تحليل هذا الرابط — تأكد من صيغته")

    # نبني المتجه بترتيب NAMES بالضبط — أي اختلاف في الترتيب يُفسد النتيجة
    vector = np.array([[feats[n] for n in NAMES]], dtype=float)

    try:
        X = SCALER.transform(vector) if SCALER is not None else vector  # التطبيع للنماذج الخطية فقط
        prob = float(MODEL.predict_proba(X)[0][1])            # احتمال التصيد
    except Exception as e:
        log.error(f"فشل التنبؤ: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="خطأ داخلي أثناء الفحص")

    is_phishing = prob >= THRESHOLD                           # مقارنة الاحتمال بالعتبة
    confidence = prob if is_phishing else 1 - prob            # الثقة في القرار المتخذ فعلاً

    # ---- تسجيل آمن -------------------------------------------------------
    # لا نسجّل الرابط الكامل أبداً: الروابط قد تحوي رموز جلسة أو معرّفات
    # حساب أو بيانات شخصية، وتسجيلها يحوّل ملف السجل نفسه إلى تسريب.
    # نسجّل بدلاً منه بصمة قصيرة (أول 10 خانات من SHA-256) تكفي لتتبع
    # طلب بعينه عند الشكوى، ولا يمكن استرجاع الرابط منها.
    fingerprint = hashlib.sha256(req.url.encode("utf-8")).hexdigest()[:10]
    log.info(f"فُحص رابط: بصمة={fingerprint} | طول={len(req.url)} "
             f"| تصيد={is_phishing} | ثقة={confidence:.3f}")

    return PredictionResponse(
        url=req.url,
        prediction="phishing" if is_phishing else "legitimate",
        label=int(is_phishing),
        confidence=round(confidence, 4),
        phishing_probability=round(prob, 4),
        threshold=THRESHOLD,
        top_features=[FeatureExplanation(**f) for f in explain(vector)],
    )


# =========================================================
# 5. معالجة أخطاء التحقق برسائل عربية مفهومة
# =========================================================
from fastapi.exceptions import RequestValidationError        # نوع خطأ التحقق في FastAPI


@app.exception_handler(RequestValidationError)
def validation_handler(request, exc):
    """يحوّل رسائل بايدانتيك التقنية إلى رسالة عربية واحدة واضحة."""
    first = exc.errors()[0] if exc.errors() else {}           # نأخذ أول خطأ فقط
    msg = first.get("msg", "مدخل غير صالح").replace("Value error, ", "")
    return JSONResponse(status_code=422, content={"detail": msg})


# =========================================================
# 6. التشغيل المباشر
# =========================================================
if __name__ == "__main__":
    import uvicorn                                            # الخادم الذي يشغّل التطبيق
    uvicorn.run(app, host="127.0.0.1", port=8000)             # محلي فقط في مرحلة التطوير
