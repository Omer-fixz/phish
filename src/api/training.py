# =========================================================
# موقع التدريب — المسارات البرمجية
# =========================================================
# صفحة مستقلة عن صفحة الفحص، مخصّصة للتدريب فقط:
#   - تعرض دراسة للخصائص الـ42 وحالة كل واحدة في البيانات
#   - تسمح باختيار الخصائص والنماذج
#   - تُشغّل التدريب في الخلفية وتعرض تقدّمه أولاً بأول
#   - تعرض النتائج، وتحفظ النموذج عند الطلب
#
# ملاحظة أمنية: هذه المسارات تُشغّل عمليات ثقيلة وتكتب ملفات نماذج،
# فهي مخصّصة للتشغيل المحلي أثناء التطوير لا للنشر العام.
# =========================================================

import datetime                                # لتسمية ملفات النماذج بالتاريخ
import shutil                                  # لنسخ النموذج عند اعتماده
import threading                               # لتشغيل التدريب دون تعليق الخادم
import traceback                               # لتسجيل تفاصيل الأخطاء

import joblib
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.paths import MODELS_DIR, MODEL_FILE, FEATURES_JSON, ROOT
from src.trainer import FEATURE_GROUPS, analyze_features, run_training

router = APIRouter(prefix="/training", tags=["training"])

TRAINING_HTML = ROOT / "src" / "api" / "static" / "training.html"


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
