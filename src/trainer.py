# =========================================================
# محرّك التدريب — المصدر الوحيد لمنطق التدريب
# =========================================================
# يستخدمه موقع التدريب (src/api/training.py) لتدريب النماذج على
# الخصائص المختارة، بنفس القواعد المنهجية المعتمدة في المشروع:
#   - التقسيم حسب النطاق (GroupShuffleSplit) لا عشوائياً
#   - التطبيع للنماذج الخطية فقط، ويُضبط على التدريب وحده
#   - العتبة تُختار من مجموعة التحقق لا من الاختبار
#   - مجموعة الاختبار تُفتح مرة واحدة في النهاية فقط
#
# الفرق عن سكربت 04_train_model.py أن هذا المحرّك:
#   - يقبل اختيار الخصائص والنماذج من المستخدم
#   - يبلّغ عن تقدّمه أولاً بأول عبر دالة رد نداء (callback)
# =========================================================

import time                                    # لقياس أزمنة التدريب
import numpy as np                             # العمليات الرقمية
import pandas as pd                            # الجداول

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from src.paths import DATASET_CSV
from src.feature_extractor import FEATURE_NAMES

# ---------------------------------------------------------
# ثوابت منهجية — مطابقة لسكربت التدريب
# ---------------------------------------------------------
SEED = 42                 # بذرة ثابتة ليتكرر نفس التقسيم في كل تشغيل
MAX_FPR = 0.10            # سقف الإنذارات الكاذبة عند اختيار العتبة
LEAKAGE_LIMIT = 0.97      # دقة أعلى من هذا تعني تسريباً لا نجاحاً


# =========================================================
# 1. تصنيف الخصائص إلى مجموعات دلالية
# =========================================================
# الترتيب هنا يطابق ترتيب FEATURE_NAMES في المستخرج.
FEATURE_GROUPS = {
    "الأطوال": FEATURE_NAMES[0:5],
    "عدّادات الرموز": FEATURE_NAMES[5:16],
    "النسب": FEATURE_NAMES[16:20],
    "خصائص ثنائية": FEATURE_NAMES[20:29],
    "خصائص بنيوية": FEATURE_NAMES[29:32],
    "خصائص دلالية": FEATURE_NAMES[32:35],
    "خصائص إحصائية": FEATURE_NAMES[35:37],
    "خصائص شبكية": FEATURE_NAMES[37:42],
}


# =========================================================
# 2. دراسة الخصائص المتاحة في البيانات
# =========================================================
def analyze_features():
    """
    يفحص ملف البيانات ويُرجع تقريراً عن كل خاصية من الخصائص الـ42:
    هل هي موجودة في البيانات؟ هل قيمتها ثابتة (ميتة)؟ وما قوة تمييزها؟

    قوة التمييز = |متوسط فئة التصيد − متوسط الفئة الشرعية| ÷ الانحراف المعياري
    كلما كبرت دلّت على قدرة أعلى على الفصل بين الفئتين.
    """
    df = pd.read_csv(DATASET_CSV, low_memory=False)
    y = df["label"]

    report = []
    for group_name, feature_list in FEATURE_GROUPS.items():
        for name in feature_list:
            entry = {"name": name, "group": group_name}

            if name not in df.columns:                    # الخاصية غير موجودة في البيانات
                entry.update({"available": False, "dead": False, "power": None,
                              "reason": "غير موجودة في ملف البيانات"})
                report.append(entry)
                continue

            col = df[name]
            unique = col.nunique()
            if unique <= 1:                               # قيمة ثابتة في كل الصفوف
                entry.update({"available": True, "dead": True, "power": 0.0,
                              "constant": float(col.iloc[0]),
                              "reason": f"قيمة ثابتة = {col.iloc[0]}"})
                report.append(entry)
                continue

            sd = float(col.std())                         # الانحراف المعياري
            power = abs(col[y == 1].mean() - col[y == 0].mean()) / sd if sd > 0 else 0.0
            entry.update({"available": True, "dead": False,
                          "power": round(float(power), 4), "reason": ""})
            report.append(entry)

    return {
        "rows": len(df),
        "domains": int(df["domain"].nunique()),
        "phishing_ratio": round(float(y.mean()), 4),
        "features": report,
        "usable": [f["name"] for f in report if f["available"] and not f["dead"]],
    }


# =========================================================
# 3. النماذج المرشحة
# =========================================================
def get_candidates(selected_families):
    """يُرجع المرشحين المطلوبين فقط: (العائلة، وصف المعاملات، النموذج، هل يحتاج تطبيعاً)."""
    all_candidates = [
        # -- خط أساس خطي: يقيس ما يمكن بلوغه بعلاقة خطية بسيطة ----------
        ("Logistic Regression", "C=0.1",
         LogisticRegression(C=0.1, max_iter=2000, random_state=SEED), True),
        ("Logistic Regression", "C=1.0",
         LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), True),

        # -- شجرة مفردة: تُظهر قدرة الشجرة وحدها قبل التجميع -------------
        ("Decision Tree", "depth=8",
         DecisionTreeClassifier(max_depth=8, random_state=SEED), False),
        ("Decision Tree", "depth=15",
         DecisionTreeClassifier(max_depth=15, random_state=SEED), False),

        # -- تجميع متوازٍ: يقلل التباين بمتوسط مئات الأشجار --------------
        ("Random Forest", "n=300",
         RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED), False),
        ("Random Forest", "n=300, depth=20",
         RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1,
                                random_state=SEED), False),

        # -- تعزيز متتابع: كل شجرة تصحح خطأ سابقتها ---------------------
        ("XGBoost", "depth=6, lr=0.1",
         XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=400,
                       n_jobs=-1, random_state=SEED, eval_metric="logloss"), False),
        ("XGBoost", "depth=10, lr=0.1",
         XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=400,
                       n_jobs=-1, random_state=SEED, eval_metric="logloss"), False),
    ]
    return [c for c in all_candidates if c[0] in selected_families]


# =========================================================
# 4. التقسيم حسب النطاق
# =========================================================
def split_by_domain(X, y, groups):
    """
    يقسّم البيانات 70/15/15 بحيث لا يظهر النطاق الواحد في أكثر من مجموعة.

    التقسيم العشوائي يضع روابط من النطاق نفسه في التدريب والاختبار،
    فيحفظ النموذج النطاق بدل أن يتعلم، وتخرج دقة متضخّمة.
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(gss2.split(X.iloc[temp_idx], y.iloc[temp_idx],
                                        groups.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    # تحقق أمني: أي تداخل في النطاقات بين المجموعات يُفسد التقييم كله
    g_tr = set(groups.iloc[train_idx])
    g_va = set(groups.iloc[val_idx])
    g_te = set(groups.iloc[test_idx])
    overlap = len(g_tr & g_va) + len(g_tr & g_te) + len(g_va & g_te)
    if overlap:
        raise RuntimeError(f"تسريب نطاقات بين المجموعات ({overlap}) — التقسيم غير سليم")

    return {
        "train": (X.iloc[train_idx], y.iloc[train_idx]),
        "val":   (X.iloc[val_idx],   y.iloc[val_idx]),
        "test":  (X.iloc[test_idx],  y.iloc[test_idx]),
    }


# =========================================================
# 5. حساب المقاييس عند عتبة محددة
# =========================================================
def evaluate(model, X, y, scaler=None, threshold=0.5):
    """يحسب كل مقاييس التقييم لنموذج على مجموعة بيانات عند عتبة معينة."""
    Xs = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(Xs)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, pred)), 4),
        "f1":        round(float(f1_score(y, pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y, proba)), 4),
    }, proba


# =========================================================
# 6. اختيار عتبة القرار من مجموعة التحقق
# =========================================================
def choose_threshold(model, parts, scaler, needs_scaling):
    """
    يختار العتبة بقاعدة تشغيلية: أعلى استدعاء بشرط FPR ≤ 10%.

    العتبة 0.5 تعامل الخطأين على أنهما متساويان، وهذا غير صحيح أمنياً:
    تفويت هجوم قد يعني سرقة بيانات، بينما الإنذار الكاذب إزعاج قابل
    للتصحيح. لكن ترجيح الاستدعاء بلا قيد يحجب روابط شرعية كثيرة
    فيفقد المستخدم ثقته بالنظام. لذلك نقيّده بسقف.

    الاختيار على التحقق لا الاختبار، وإلا فقد الرقم النهائي معناه.
    """
    X_va, y_va = parts["val"]
    Xs = scaler.transform(X_va) if needs_scaling else X_va
    proba = model.predict_proba(Xs)[:, 1]

    curve, best_thr, best_recall = [], 0.5, -1
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= thr).astype(int)
        p = precision_score(y_va, pred, zero_division=0)
        r = recall_score(y_va, pred)
        fpr = ((pred == 1) & (y_va.values == 0)).sum() / max((y_va.values == 0).sum(), 1)
        curve.append({"threshold": round(float(thr), 2), "precision": round(float(p), 4),
                      "recall": round(float(r), 4), "fpr": round(float(fpr), 4)})
        if fpr <= MAX_FPR and r > best_recall:
            best_thr, best_recall = round(float(thr), 2), r

    return best_thr, curve


# =========================================================
# 7. التدريب الكامل
# =========================================================
def run_training(features, families, progress):
    """
    يُجري دورة تدريب كاملة على الخصائص والنماذج المختارة.

    المعاملات:
        features : قائمة أسماء الخصائص المراد التدريب عليها
        families : قائمة عائلات النماذج المطلوبة
        progress : دالة رد نداء progress(percent, message) لتحديث الواجهة

    المخرجات: قاموس فيه جدول المقارنة والنتيجة النهائية وحزمة النموذج.
    """
    progress(2, "قراءة البيانات...")
    df = pd.read_csv(DATASET_CSV, low_memory=False)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"خصائص مطلوبة غير موجودة في البيانات: {missing}")
    if not features:
        raise ValueError("لم تُختَر أي خاصية")
    if not families:
        raise ValueError("لم يُختَر أي نموذج")

    X, y, groups = df[features], df["label"].astype(int), df["domain"]

    progress(6, f"التقسيم حسب النطاق ({len(X)} رابطاً، {groups.nunique()} نطاقاً)...")
    parts = split_by_domain(X, y, groups)

    # التطبيع يُضبَط على التدريب وحده حتى لا تتسرب إحصاءات التحقق والاختبار
    scaler = StandardScaler().fit(parts["train"][0])

    candidates = get_candidates(families)
    if not candidates:
        raise ValueError("لا يوجد مرشحون مطابقون للاختيار")

    X_tr, y_tr = parts["train"]
    results = []

    # ---- تدريب كل مرشّح وقياسه على مجموعة التحقق ----------------------
    for i, (family, params, model, needs_scaling) in enumerate(candidates, start=1):
        pct = 8 + int(72 * (i - 1) / len(candidates))
        progress(pct, f"تدريب {family} ({params}) — {i} من {len(candidates)}")

        t0 = time.time()
        Xt = scaler.transform(X_tr) if needs_scaling else X_tr
        model.fit(Xt, y_tr)
        secs = round(time.time() - t0, 1)

        m, _ = evaluate(model, *parts["val"], scaler if needs_scaling else None, 0.5)
        results.append({"family": family, "params": params, "scaled": needs_scaling,
                        "seconds": secs, "model": model, **m})
        progress(pct, f"{family} ({params}) → F1 = {m['f1']}  في {secs}ث")

    # ---- الأفضل من كل عائلة حسب F1 على التحقق -------------------------
    best_per_family = {}
    for r in results:
        if r["family"] not in best_per_family or r["f1"] > best_per_family[r["family"]]["f1"]:
            best_per_family[r["family"]] = r

    winner = max(best_per_family.values(), key=lambda r: r["f1"])
    progress(82, f"النموذج الفائز: {winner['family']} ({winner['params']})")

    # ---- اختيار العتبة من مجموعة التحقق -------------------------------
    progress(86, "اختيار عتبة القرار من مجموعة التحقق...")
    threshold, curve = choose_threshold(winner["model"], parts, scaler, winner["scaled"])

    # ---- الاختبار النهائي: تُفتح مجموعة الاختبار مرة واحدة فقط --------
    progress(92, f"التقييم النهائي على مجموعة الاختبار عند عتبة {threshold}...")
    sc = scaler if winner["scaled"] else None
    test_metrics, proba = evaluate(winner["model"], *parts["test"], sc, threshold)

    X_te, y_te = parts["test"]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    # ---- قياس أثر التقسيم العشوائي (لتوثيق حجم التسريب) ---------------
    progress(96, "قياس أثر التقسيم العشوائي للمقارنة...")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.15,
                                                  random_state=SEED, stratify=y)
    clone = winner["model"].__class__(**winner["model"].get_params())
    clone.fit(scaler.transform(Xr_tr) if winner["scaled"] else Xr_tr, yr_tr)
    Xr_te_s = scaler.transform(Xr_te) if winner["scaled"] else Xr_te
    random_acc = round(float(accuracy_score(yr_te, clone.predict(Xr_te_s))), 4)

    # ---- أهمية الخصائص ------------------------------------------------
    importance = []
    if hasattr(winner["model"], "feature_importances_"):
        imp = pd.Series(winner["model"].feature_importances_, index=features)
        importance = [{"name": n, "value": round(float(v), 4)}
                      for n, v in imp.sort_values(ascending=False).head(15).items()]

    progress(99, "تجهيز النتائج...")

    bundle = {
        "model": winner["model"],
        "scaler": scaler if winner["scaled"] else None,
        "threshold": threshold,
        "feature_names": list(features),
        "model_name": winner["family"],
        "params": winner["params"],
        "test_metrics": test_metrics,
    }

    return {
        "dataset": {"rows": len(df), "domains": int(groups.nunique()),
                    "features": len(features),
                    "train": len(parts["train"][0]), "val": len(parts["val"][0]),
                    "test": len(parts["test"][0])},
        "comparison": [{k: v for k, v in r.items() if k != "model"}
                       for r in sorted(best_per_family.values(),
                                       key=lambda r: r["f1"], reverse=True)],
        "all_runs": [{k: v for k, v in r.items() if k != "model"} for r in results],
        "winner": {"family": winner["family"], "params": winner["params"]},
        "threshold": threshold,
        "threshold_curve": curve,
        "test_metrics": test_metrics,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "error_rates": {"fnr": round(float(fn / max(fn + tp, 1)), 4),
                        "fpr": round(float(fp / max(tn + fp, 1)), 4)},
        "random_split_accuracy": random_acc,
        "leak_gap": round(random_acc - test_metrics["accuracy"], 4),
        "leakage_warning": test_metrics["accuracy"] > LEAKAGE_LIMIT,
        "importance": importance,
        "_bundle": bundle,
    }
