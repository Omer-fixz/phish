# =========================================================
# موقع التدريب — المسارات البرمجية
# =========================================================
# صفحة مستقلة عن صفحة الفحص، مخصّصة للتدريب فقط:
#   - تعرض دراسة للخصائص الـ42 وحالة كل واحدة في البيانات
#   - تسمح باختيار الخصائص والنماذج
#   - تُشغّل التدريب في الخلفية وتعرض تقدّمه أولاً بأول
#   - تعرض النتائج، وتحفظ النموذج عند الطلب
#
# =========================================================
# تنبيه أمني — سبب وجود مفتاح التفعيل أدناه
# =========================================================
# هذه المسارات تُشغّل تدريباً يستهلك المعالج دقائق، وتكتب ملفات نماذج،
# ويمكنها استبدال النموذج الذي يعمل به مسار الفحص. ولا تخضع لتحديد
# معدل الطلبات لأن التدريب عملية طويلة بطبيعتها.
#
# لو نُشرت على خادم عام لاستطاع أي زائر أن يستهلك المعالج أو يستبدل
# النموذج. لذلك لا تُسجَّل هذه المسارات إطلاقاً ما لم يُضبط متغير
# البيئة ENABLE_TRAINING=1.
#
# التقسيم العملي:
#   - محلياً: run_api.py يضبط المتغير تلقائياً، فالموقع يعمل.
#   - عند النشر: Procfile يشغّل uvicorn مباشرة بلا المتغير،
#     فالمسارات غير موجودة أصلاً — لا 403 ولا صفحة تُخبر بوجودها.
# =========================================================

import datetime                                # لتسمية ملفات النماذج بالتاريخ
import os                                      # لقراءة متغير البيئة
import shutil                                  # لنسخ النموذج عند اعتماده
import threading                               # لتشغيل التدريب دون تعليق الخادم
import traceback                               # لتسجيل تفاصيل الأخطاء

import joblib
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.paths import MODELS_DIR, MODEL_FILE, FEATURES_JSON, ROOT
from src.trainer import (FEATURE_GROUPS, analyze_features, run_training,
                         add_custom_links, custom_summary, clear_custom_links)

router = APIRouter(prefix="/training", tags=["training"])

TRAINING_HTML = ROOT / "src" / "api" / "static" / "training.html"

def is_enabled(value: str | None) -> bool:
    """
    يقرر هل يُفعَّل موقع التدريب بناءً على قيمة متغير البيئة.

    الوضع الآمن هو الافتراضي: أي قيمة غير معروفة — وكذلك غياب
    المتغير تماماً — تعني الإغلاق. فُصلت في دالة مستقلة ليكون
    منطق القرار قابلاً للاختبار دون تشغيل خادم.
    """
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# مفتاح التفعيل يُقرأ مرة واحدة عند الاستيراد، لأن تسجيل المسارات
# يحدث وقت الاستيراد ولا يمكن تغييره بعد إقلاع الخادم.
ENABLED = is_enabled(os.getenv("ENABLE_TRAINING"))


# =========================================================
# 1. حالة المهمة الجارية
# =========================================================
# نحتفظ بحالة واحدة فقط لأن التدريب عملية ثقيلة لا يصح تشغيل أكثر
# من واحدة منها في وقت واحد على نفس الجهاز.
_job = {
    "running": False,      # هل يوجد تدريب جارٍ الآن؟
    "percent": 0,          # نسبة الإنجاز
    "message": "جاهز",     # آخر رسالة حالة
    "log": [],             # سجل الخطوات لعرضه في الواجهة
    "result": None,        # النتائج بعد الانتهاء
    "error": None,         # نص الخطأ إن فشل
    "started_at": None,    # وقت البدء
}
_lock = threading.Lock()   # يمنع تعديل الحالة من خيطين في وقت واحد
_last_bundle = {"data": None}   # حزمة آخر نموذج مدرَّب (لا تُحفظ إلا بطلب)


def _progress(percent, message):
    """تُستدعى من محرّك التدريب لتحديث الحالة، فتظهر في الواجهة فوراً."""
    with _lock:
        _job["percent"] = percent
        _job["message"] = message
        _job["log"].append({"percent": percent, "message": message})


# =========================================================
# 2. شكل الطلب
# =========================================================
class TrainRequest(BaseModel):
    """اختيارات المستخدم: أي خصائص وأي نماذج."""
    features: list[str] = Field(..., min_length=1,
                                description="أسماء الخصائص المراد التدريب عليها")
    families: list[str] = Field(..., min_length=1,
                                description="عائلات النماذج المطلوبة")


class SaveRequest(BaseModel):
    """طلب حفظ النموذج المدرَّب."""
    promote: bool = Field(False, description="هل يُعتمد نموذجاً رسمياً للنظام؟")


class EvaluateRequest(BaseModel):
    """
    طلب فحص دفعة روابط بالنموذج المدرَّب.

    يقبل الرابط وحده، أو الرابط ومعه تصنيفه الحقيقي مفصولين بفاصلة
    (مثل صيغة CSV: `example.com/login,1`). فإن وُجدت التصنيفات
    حُسبت الدقة أيضاً، وإلا اكتُفي بعرض التنبؤات.
    """
    urls: list[str] = Field(..., min_length=1, max_length=5000,
                            description="الروابط، رابط في كل عنصر")


# =========================================================
# 3. المسارات
# =========================================================

@router.get("", response_class=HTMLResponse)
def training_page():
    """يقدّم صفحة موقع التدريب."""
    if not TRAINING_HTML.exists():
        raise HTTPException(status_code=404, detail="training.html غير موجود")
    return HTMLResponse(TRAINING_HTML.read_text(encoding="utf-8"))


@router.get("/features")
def features_catalog():
    """
    دراسة الخصائص: يُرجع كل خاصية مع مجموعتها وحالتها في البيانات.

    هذا المسار هو ما يجيب عن سؤال «ما الخصائص المتاحة فعلاً للتدريب؟»
    قبل بدء أي تدريب.
    """
    try:
        info = analyze_features()
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                            detail="ملف البيانات غير موجود — شغّل scripts/01_build_dataset.py")
    info["groups"] = {name: list(feats) for name, feats in FEATURE_GROUPS.items()}
    return info


@router.get("/status")
def status():
    """حالة التدريب الجارية — تستعلم عنها الواجهة كل ثانية."""
    with _lock:
        return {k: v for k, v in _job.items() if k != "result"} | {
            "has_result": _job["result"] is not None,
            "result": _job["result"],
        }


@router.post("/start")
def start(req: TrainRequest):
    """يبدأ تدريباً جديداً في خيط منفصل حتى لا يتجمّد الخادم."""
    with _lock:
        if _job["running"]:
            raise HTTPException(status_code=409, detail="يوجد تدريب جارٍ بالفعل")
        _job.update({"running": True, "percent": 0, "message": "بدء التشغيل...",
                     "log": [], "result": None, "error": None,
                     "started_at": datetime.datetime.now().strftime("%H:%M:%S")})

    def worker():
        """الخيط الذي ينفّذ التدريب فعلياً."""
        try:
            out = run_training(req.features, req.families, _progress)
            _last_bundle["data"] = out.pop("_bundle")     # نحتفظ بها للحفظ لاحقاً
            with _lock:
                _job.update({"result": out, "percent": 100, "message": "اكتمل التدريب"})
        except Exception as e:
            traceback.print_exc()
            with _lock:
                _job.update({"error": f"{type(e).__name__}: {e}",
                             "message": "فشل التدريب"})
        finally:
            with _lock:
                _job["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}


class AddDataRequest(BaseModel):
    """
    روابط يضيفها المستخدم إلى بيانات التدريب.

    التصنيف إلزامي مع كل رابط بصيغة `الرابط,1` أو `الرابط,0`،
    لأن التدريب بلا تصنيف مستحيل.
    """
    lines: list[str] = Field(..., min_length=1, max_length=20000,
                             description="سطر لكل رابط بصيغة url,label")


@router.get("/data")
def get_custom_data():
    """ملخّص بيانات التدريب الإضافية المحفوظة حالياً."""
    return custom_summary()


@router.post("/data")
def post_custom_data(req: AddDataRequest):
    """
    يضيف روابط المستخدم إلى بيانات التدريب.

    الاستخراج بلا اتصال بالشبكة، فإضافة آلاف الروابط تستغرق ثوانٍ.
    """
    result = add_custom_links(req.lines)
    result["summary"] = custom_summary()
    return result


@router.delete("/data")
def delete_custom_data():
    """يحذف كل بيانات التدريب الإضافية ويُبقي الأصل سليماً."""
    return {"deleted": clear_custom_links()}


@router.post("/evaluate")
def evaluate_batch(req: EvaluateRequest):
    """
    يفحص دفعة روابط بالنموذج المدرَّب حديثاً (أو الرسمي إن لم يُدرَّب شيء).

    الغرض: تجربة النموذج فور تدريبه على روابط تختارها بنفسك، بدل
    الاكتفاء بأرقام مجموعة الاختبار.

    ملاحظة أمنية: الروابط تُعامَل كنصوص فقط ولا تُفتح إطلاقاً.
    """
    import numpy as np
    from src.feature_extractor import extract_features

    bundle = _last_bundle["data"]
    source = "النموذج المدرَّب في هذه الجلسة"
    if bundle is None:                                   # لم يُدرَّب شيء بعد
        if not MODEL_FILE.exists():
            raise HTTPException(status_code=400, detail="لا يوجد نموذج — درّب أولاً")
        bundle = joblib.load(MODEL_FILE)
        source = "النموذج الرسمي المحفوظ"

    names = bundle["feature_names"]
    model, scaler = bundle["model"], bundle.get("scaler")
    threshold = float(bundle.get("threshold", 0.5))

    # ---- تفكيك المدخلات: رابط، أو "رابط,تصنيف" ----------------------
    rows, truth = [], []
    for raw in req.urls:
        line = raw.strip().strip('"').strip("'")
        if not line:
            continue
        label = None
        if "," in line:                                  # صيغة CSV بسيطة
            head, _, tail = line.rpartition(",")
            if tail.strip() in ("0", "1") and head.strip():
                line, label = head.strip(), int(tail.strip())
        rows.append(line)
        truth.append(label)

    if not rows:
        raise HTTPException(status_code=400, detail="لا يوجد رابط صالح في المدخلات")

    # ---- الاستخراج والتنبؤ -------------------------------------------
    vectors, failed = [], []
    for u in rows:
        try:
            f = extract_features(u)
            vectors.append([f[n] for n in names])
        except Exception:
            vectors.append([0] * len(names))
            failed.append(u)

    X = np.array(vectors, dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]

    results = []
    for u, p, t in zip(rows, proba, truth):
        is_phish = bool(p >= threshold)
        results.append({
            "url": u,
            "probability": round(float(p), 4),
            "prediction": "phishing" if is_phish else "legitimate",
            "label": int(is_phish),
            "truth": t,
            "correct": None if t is None else (int(is_phish) == t),
        })

    # ---- ملخص، مع الدقة إن توفّرت التصنيفات الحقيقية ------------------
    labeled = [r for r in results if r["truth"] is not None]
    summary = {
        "total": len(results),
        "phishing": sum(r["label"] for r in results),
        "legitimate": sum(1 - r["label"] for r in results),
        "failed": len(failed),
        "model_source": source,
        "threshold": threshold,
        "labeled": len(labeled),
    }
    if labeled:
        correct = sum(1 for r in labeled if r["correct"])
        pos = [r for r in labeled if r["truth"] == 1]
        neg = [r for r in labeled if r["truth"] == 0]
        summary["accuracy"] = round(correct / len(labeled), 4)
        if pos:
            summary["recall"] = round(sum(1 for r in pos if r["label"] == 1) / len(pos), 4)
        if neg:
            summary["fpr"] = round(sum(1 for r in neg if r["label"] == 1) / len(neg), 4)

    return {"summary": summary, "results": results}


@router.post("/save")
def save(req: SaveRequest):
    """
    يحفظ النموذج المدرَّب.

    يُحفظ دائماً باسم مؤرَّخ في models/. أما اعتماده نموذجاً رسمياً
    للنظام (model.joblib) فلا يحدث إلا بطلب صريح، حتى لا يُستبدل
    النموذج العامل بالخطأ.
    """
    bundle = _last_bundle["data"]
    if bundle is None:
        raise HTTPException(status_code=400, detail="لا يوجد نموذج مدرَّب لحفظه")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"trained_{stamp}.joblib"
    joblib.dump(bundle, path)

    result = {"saved": str(path.relative_to(ROOT)), "promoted": False}

    if req.promote:
        # نأخذ نسخة احتياطية من النموذج الحالي قبل استبداله
        if MODEL_FILE.exists():
            backup = MODELS_DIR / f"model_backup_{stamp}.joblib"
            shutil.copy2(MODEL_FILE, backup)
            result["backup"] = str(backup.relative_to(ROOT))

        joblib.dump(bundle, MODEL_FILE)
        FEATURES_JSON.write_text(
            __import__("json").dumps(bundle["feature_names"], ensure_ascii=False, indent=2),
            encoding="utf-8")
        result["promoted"] = True
        result["note"] = "أعد تشغيل الخادم ليحمّل النموذج الجديد في مسار الفحص"

    return result
