# =========================================================
# المرحلة 2 — التدريب والمقارنة بين النماذج
# =========================================================
# ما الذي يفعله هذا الملف؟
#   1. يقسّم البيانات حسب النطاق (لا عشوائياً) إلى تدريب/تحقق/اختبار
#   2. يدرّب أربعة نماذج ويضبط معاملاتها على مجموعة التحقق
#   3. يختار الأفضل ويفتح مجموعة الاختبار مرة واحدة فقط في النهاية
#   4. يحفظ النموذج مع أسماء الخصائص بترتيبها الثابت
# =========================================================

import sys                          # للتحكم في إعدادات بايثون
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # لعرض العربية في الشاشة بلا أخطاء

import os                           # للتعامل مع المجلدات
import json                         # لحفظ أسماء الخصائص في ملف JSON
import time                         # لقياس زمن التدريب
import numpy as np                  # العمليات الرقمية
import pandas as pd                 # الجداول
import joblib                       # لحفظ النموذج المدرَّب في ملف

import matplotlib                   # مكتبة الرسم
matplotlib.use("Agg")               # الرسم في ملفات فقط بدون فتح نوافذ
import matplotlib.pyplot as plt     # واجهة الرسم
import seaborn as sns               # رسومات أجمل

from sklearn.model_selection import GroupShuffleSplit, train_test_split  # أدوات التقسيم
from sklearn.preprocessing import StandardScaler                          # التطبيع للنماذج الخطية
from sklearn.linear_model import LogisticRegression                       # النموذج 1: انحدار لوجستي
from sklearn.tree import DecisionTreeClassifier                           # النموذج 2: شجرة قرار
from sklearn.ensemble import RandomForestClassifier                       # النموذج 3: غابة عشوائية
from xgboost import XGBClassifier                                         # النموذج 4: XGBoost
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)                                   # مقاييس التقييم

# ---------------------------------------------------------
# إعدادات عامة في مكان واحد
# ---------------------------------------------------------
DATA_FILE = "dataset_v2.csv"      # ملف البيانات المبني بالمستخرج الرسمي
FIG_DIR = "figures"               # مجلد الرسومات
MODEL_FILE = "model.joblib"       # اسم ملف النموذج النهائي (اسم محايد لأن الفائز قد يكون أي نموذج)
FEATURES_FILE = "feature_names.json"  # ملف أسماء الخصائص بترتيبها
SEED = 42                         # رقم ثابت ليتكرر نفس التقسيم في كل تشغيل
LEAKAGE_LIMIT = 0.97              # أي دقة أعلى من هذا تعني تسريباً وليس نجاحاً
MAX_FPR = 0.10                    # أقصى نسبة إنذارات كاذبة مقبولة عند اختيار العتبة

sns.set_theme(style="whitegrid")  # شكل موحّد للرسومات


# =========================================================
# 1. تحميل البيانات وتجهيزها
# =========================================================
def load_data():
    """يقرأ الملف ويفصل الخصائص عن التصنيف عن النطاقات، ويحذف الخصائص الميتة."""
    df = pd.read_csv(DATA_FILE, low_memory=False)              # قراءة الملف كاملاً
    print(f"✅ البيانات: {df.shape[0]} صف × {df.shape[1]} عمود")

    y = df["label"].astype(int)                                # الهدف: 1 تصيد / 0 شرعي
    groups = df["domain"]                                      # المجموعات: النطاق، وعليه سيتم التقسيم
    X = df.drop(columns=["url", "domain", "label"])            # الخصائص فقط (بلا نص ولا هدف)

    # الخاصية التي قيمتها ثابتة في كل الصفوف لا تحمل أي معلومة، فنحذفها
    dead = [c for c in X.columns if X[c].nunique() <= 1]
    if dead:
        X = X.drop(columns=dead)                               # الحذف الفعلي
        print(f"🪦 حُذفت خصائص ميتة: {dead}")
    print(f"📐 عدد الخصائص المستخدمة = {X.shape[1]}")

    return X, y, groups


# =========================================================
# 2. التقسيم حسب النطاق (70% / 15% / 15%)
# =========================================================
def split_by_domain(X, y, groups):
    """
    يقسّم البيانات بحيث لا يظهر النطاق الواحد في أكثر من مجموعة.

    لماذا لا نقسّم عشوائياً؟
    73% من الروابط تقع في نطاقات مكررة. التقسيم العشوائي يضع روابط
    من نفس النطاق في التدريب والاختبار، فيحفظ النموذج النطاق بدل أن
    يتعلم شكل الرابط الخبيث، وتخرج دقة متضخّمة لا تتكرر في الواقع.
    """
    # الخطوة الأولى: نفصل 30% مؤقتة (ستُقسم لاحقاً إلى تحقق واختبار)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups))       # الحصول على أرقام صفوف كل جزء

    # الخطوة الثانية: نقسم الـ30% إلى نصفين متساويين = 15% تحقق + 15% اختبار
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(gss2.split(X.iloc[temp_idx], y.iloc[temp_idx],
                                        groups.iloc[temp_idx]))
    val_idx = temp_idx[val_rel]                                # تحويل الأرقام النسبية إلى أرقام أصلية
    test_idx = temp_idx[test_rel]

    parts = {
        "train": (X.iloc[train_idx], y.iloc[train_idx]),       # مجموعة التدريب
        "val":   (X.iloc[val_idx],   y.iloc[val_idx]),         # مجموعة التحقق (لضبط المعاملات)
        "test":  (X.iloc[test_idx],  y.iloc[test_idx]),        # مجموعة الاختبار (تُفتح مرة واحدة فقط)
    }

    total = len(X)                                             # الإجمالي لحساب النسب
    for name, (Xp, yp) in parts.items():                       # طباعة حجم كل مجموعة ونسبتها
        print(f"   {name:5s}: {len(Xp):6d} صف = {len(Xp)/total*100:4.1f}%  "
              f"| نسبة التصيد فيها = {yp.mean()*100:4.1f}%")

    # تحقق أمني: يجب ألا يشترك أي نطاق بين أي مجموعتين
    g_tr, g_va, g_te = (set(groups.iloc[i]) for i in (train_idx, val_idx, test_idx))
    overlap = len(g_tr & g_va) + len(g_tr & g_te) + len(g_va & g_te)
    print(f"   🔒 نطاقات مشتركة بين المجموعات = {overlap} (يجب أن تكون صفراً)")
    if overlap:                                                # لو حدث تداخل فالتقسيم فاسد
        raise RuntimeError("تسريب نطاقات بين المجموعات — التقسيم غير سليم")

    return parts


# =========================================================
# 3. حساب المقاييس
# =========================================================
def evaluate(model, X, y, scaler=None, threshold=0.5):
    """يحسب كل مقاييس التقييم لنموذج على مجموعة بيانات معينة عند عتبة محددة."""
    Xs = scaler.transform(X) if scaler is not None else X       # نطبّع فقط إن كان النموذج خطياً
    proba = model.predict_proba(Xs)[:, 1]                       # احتمال أن يكون الرابط تصيداً
    pred = (proba >= threshold).astype(int)                     # القرار: أعلى من العتبة = تصيد

    return {
        "accuracy":  accuracy_score(y, pred),                   # نسبة التنبؤات الصحيحة إجمالاً
        "precision": precision_score(y, pred),                  # من قلنا عنها تصيد، كم منها تصيد فعلاً
        "recall":    recall_score(y, pred),                     # من روابط التصيد الحقيقية، كم أمسكنا
        "f1":        f1_score(y, pred),                         # المتوسط التوافقي بين الدقة والاستدعاء
        "roc_auc":   roc_auc_score(y, proba),                   # قدرة النموذج على الترتيب بين الفئتين
        "_pred": pred, "_proba": proba,                         # نحتفظ بها للرسومات لاحقاً
    }


# =========================================================
# 4. شبكات المعاملات لكل نموذج
# =========================================================
# ملاحظة منهجية: لم نستخدم StratifiedKFold للضبط لأن الطي العشوائي
# داخل بيانات التدريب يخلط نطاقات الطية الواحدة مع الأخرى فيعيد
# نفس التسريب الذي تجنّبناه بالتقسيم حسب النطاق. البديل الأنظف:
# ندرّب كل مرشّح على التدريب ونقيسه على التحقق، ومجموعة التحقق
# منفصلة تماماً بالنطاق.
def get_candidates():
    """يُرجع قائمة النماذج المرشحة: (اسم العائلة، وصف المعاملات، النموذج، هل يحتاج تطبيعاً)."""
    return [
        # -- النموذج 1: الانحدار اللوجستي (خطي، يحتاج تطبيعاً) --------------
        ("Logistic Regression", "C=0.1",
         LogisticRegression(C=0.1, max_iter=2000, random_state=SEED), True),
        ("Logistic Regression", "C=1.0",
         LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), True),

        # -- النموذج 2: شجرة قرار واحدة (بسيطة وسهلة التفسير) --------------
        ("Decision Tree", "depth=8",
         DecisionTreeClassifier(max_depth=8, random_state=SEED), False),
        ("Decision Tree", "depth=15",
         DecisionTreeClassifier(max_depth=15, random_state=SEED), False),
        ("Decision Tree", "depth=None",
         DecisionTreeClassifier(random_state=SEED), False),

        # -- النموذج 3: الغابة العشوائية (مئات الأشجار مجتمعة) -------------
        ("Random Forest", "n=300, depth=None",
         RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED), False),
        ("Random Forest", "n=300, depth=20",
         RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1,
                                random_state=SEED), False),
        ("Random Forest", "n=500, leaf=3",
         RandomForestClassifier(n_estimators=500, min_samples_leaf=3, n_jobs=-1,
                                random_state=SEED), False),

        # -- النموذج 4: XGBoost (أشجار متتابعة يصحح بعضها بعضاً) ----------
        ("XGBoost", "depth=6, lr=0.1",
         XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=400,
                       n_jobs=-1, random_state=SEED, eval_metric="logloss"), False),
        ("XGBoost", "depth=10, lr=0.1",
         XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=400,
                       n_jobs=-1, random_state=SEED, eval_metric="logloss"), False),
    ]


# =========================================================
# 5. تدريب كل المرشحين واختيار الأفضل في كل عائلة
# =========================================================
def train_all(parts, scaler):
    """يدرّب كل مرشّح على التدريب ويقيسه على التحقق، ثم يُبقي الأفضل من كل عائلة."""
    X_tr, y_tr = parts["train"]                                 # بيانات التدريب
    X_va, y_va = parts["val"]                                   # بيانات التحقق

    results = []                                                # كل النتائج للجدول التفصيلي
    print("\n🏋️ تدريب المرشحين (القياس على مجموعة التحقق):")

    for family, params, model, needs_scaling in get_candidates():
        t0 = time.time()                                        # بداية التوقيت
        Xt = scaler.transform(X_tr) if needs_scaling else X_tr  # التطبيع للنماذج الخطية فقط
        model.fit(Xt, y_tr)                                     # التدريب الفعلي
        m = evaluate(model, X_va, y_va, scaler if needs_scaling else None)  # القياس على التحقق
        secs = time.time() - t0                                 # زمن التدريب

        results.append({"family": family, "params": params, "model": model,
                        "scaled": needs_scaling, "time": secs, **m})
        print(f"   {family:20s} {params:18s} F1={m['f1']:.4f}  "
              f"Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}  ({secs:.1f}ث)")

    # من كل عائلة نُبقي المرشّح صاحب أعلى F1 على التحقق
    best_per_family = {}
    for r in results:
        if r["family"] not in best_per_family or r["f1"] > best_per_family[r["family"]]["f1"]:
            best_per_family[r["family"]] = r

    return results, best_per_family


# =========================================================
# 6. جدول المقارنة النهائي بين النماذج الأربعة
# =========================================================
def comparison_table(best_per_family):
    """يبني ويطبع جدول المقارنة بين أفضل نسخة من كل نموذج."""
    rows = []                                                   # صفوف الجدول
    for family, r in best_per_family.items():
        rows.append({
            "النموذج": family, "المعاملات": r["params"],
            "Accuracy": round(r["accuracy"], 4), "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4), "F1": round(r["f1"], 4),
            "ROC-AUC": round(r["roc_auc"], 4), "الزمن (ث)": round(r["time"], 1),
        })

    table = pd.DataFrame(rows).sort_values("F1", ascending=False)  # ترتيب تنازلي حسب F1
    print("\n📊 جدول المقارنة (على مجموعة التحقق):")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(FIG_DIR, "model_comparison.csv"), index=False,
                 encoding="utf-8-sig")                          # utf-8-sig ليفتح صحيحاً في إكسل
    print(f"   💾 حُفظ الجدول في {FIG_DIR}/model_comparison.csv")

    # رسم بياني يقارن F1 بين النماذج الأربعة
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=table, x="F1", y="النموذج", hue="النموذج",
                palette="viridis", legend=False)
    plt.title("Model Comparison — F1 on Validation Set")        # عنوان إنجليزي لأن matplotlib لا يعرض العربية
    plt.xlabel("F1 Score")
    plt.ylabel("")
    plt.xlim(0.7, 1.0)                                          # تقريب المدى لإبراز الفروق الصغيرة
    save_fig("06_model_comparison.png")

    return table


# =========================================================
# 6ب. اختيار عتبة القرار من مجموعة التحقق
# =========================================================
def choose_threshold(winner, parts, scaler):
    """
    يختار عتبة القرار من البيانات بدل كتابة رقم يدوياً.

    لماذا لا نترك 0.5؟
    العتبة 0.5 تعامل الخطأين على أنهما متساويان، وهذا غير صحيح أمنياً:
    تفويت هجوم تصيد (FN) يعني سرقة بيانات المستخدم فعلاً، بينما الإنذار
    الكاذب (FP) يعني إزعاجاً قابلاً للتصحيح بنظرة من المستخدم.
    لكن ترجيح الاستدعاء وحده لا يكفي: تجربة مقياس F2 على هذه البيانات
    اختارت عتبة 0.20 التي تحجب 21% من الروابط الشرعية — رقم يُفقد
    المستخدم ثقته في النظام فيتجاهل تحذيراته كلها، فتضيع فائدة النظام.

    لذلك نستخدم قاعدة تشغيلية مقيَّدة، وهي المتبعة في أنظمة الأمن:
        «أعلى استدعاء ممكن، بشرط ألا تتجاوز الإنذارات الكاذبة سقفاً محدداً»
    السقف هنا MAX_FPR أي 10% من الروابط الشرعية كحد أقصى.

    مهم: الاختيار يتم على التحقق لا على الاختبار، وإلا صار ضبطاً
    على البيانات التي نُقيَّم بها وفقدت النتيجة النهائية معناها.
    """
    X_va, y_va = parts["val"]                                    # مجموعة التحقق
    sc = scaler if winner["scaled"] else None                    # المطبّع إن لزم
    Xs = sc.transform(X_va) if sc is not None else X_va
    proba = winner["model"].predict_proba(Xs)[:, 1]              # احتمالات التصيد

    print("\n🎚️ اختيار عتبة القرار على مجموعة التحقق:")
    print(f"   القاعدة: أعلى Recall بشرط FPR ≤ {MAX_FPR:.0%}")
    print(f"   {'العتبة':>8} {'Precision':>10} {'Recall':>8} {'FPR':>8} {'F1':>8}")

    best_thr, best_recall = 0.5, -1                              # أفضل عتبة وأفضل استدعاء ضمن الشرط
    for thr in np.arange(0.05, 0.96, 0.01):                      # مسح دقيق للعتبات
        pred = (proba >= thr).astype(int)                        # القرار عند هذه العتبة
        p = precision_score(y_va, pred, zero_division=0)          # الدقة
        r = recall_score(y_va, pred)                              # الاستدعاء
        # نسبة الإنذارات الكاذبة = كم رابطاً شرعياً حُجب من كل الروابط الشرعية
        fpr = ((pred == 1) & (y_va.values == 0)).sum() / (y_va.values == 0).sum()

        if round(thr, 2) in (0.20, 0.35, 0.50, 0.65, 0.80):       # نطبع عيّنة فقط لا كل الصفوف
            print(f"   {thr:8.2f} {p:10.4f} {r:8.4f} {fpr:8.4f} {f1_score(y_va, pred):8.4f}")

        if fpr <= MAX_FPR and r > best_recall:                    # ضمن السقف وأفضل استدعاء
            best_thr, best_recall = round(float(thr), 2), r

    print(f"   ➜ العتبة المختارة = {best_thr} (استدعاء {best_recall:.4f} ضمن سقف الإنذارات)")
    print("     تُحفظ داخل النموذج ويقرأها التطبيق تلقائياً — لا رقم مكتوب يدوياً.")
    return best_thr


# =========================================================
# 7. تقييم النموذج الفائز على مجموعة الاختبار (مرة واحدة فقط)
# =========================================================
def final_test(winner, parts, scaler, X_columns, threshold):
    """يفتح مجموعة الاختبار مرة واحدة ويُخرج التقييم النهائي الكامل."""
    X_te, y_te = parts["test"]                                  # مجموعة الاختبار
    sc = scaler if winner["scaled"] else None                   # نمرر المطبّع فقط إن كان النموذج خطياً
    m = evaluate(winner["model"], X_te, y_te, sc, threshold)    # حساب كل المقاييس عند العتبة المختارة

    print("\n" + "=" * 60)
    print(f"النتيجة النهائية — {winner['family']} ({winner['params']}) عند عتبة {threshold}")
    print("=" * 60)
    print(f"   Accuracy  = {m['accuracy']:.4f}")                # نسبة الصواب الإجمالية
    print(f"   Precision = {m['precision']:.4f}")               # نقاء إنذارات التصيد
    print(f"   Recall    = {m['recall']:.4f}")                  # نسبة هجمات التصيد الملتقطة
    print(f"   F1        = {m['f1']:.4f}")
    print(f"   ROC-AUC   = {m['roc_auc']:.4f}")

    # ---- مصفوفة الالتباس ----------------------------------------------
    cm = confusion_matrix(y_te, m["_pred"])                      # جدول 2×2 للصواب والخطأ
    tn, fp, fn, tp = cm.ravel()                                  # تفكيكها إلى أربعة أرقام
    print(f"\n   مصفوفة الالتباس:")
    print(f"      شرعي صُنّف شرعي  (TN) = {tn}")
    print(f"      شرعي صُنّف تصيد  (FP) = {fp}  ← إنذار كاذب")
    print(f"      تصيد صُنّف شرعي  (FN) = {fn}  ← هجوم فائت")
    print(f"      تصيد صُنّف تصيد  (TP) = {tp}")

    # ---- المفاضلة بين نوعي الخطأ ---------------------------------------
    print(f"\n   ⚖️ المفاضلة بين الخطأين:")
    print(f"      • هجوم فائت (FN): {fn} رابط تصيد مرّ كآمن = {fn/(fn+tp)*100:.1f}% من هجمات التصيد.")
    print(f"        تكلفته سرقة بيانات المستخدم فعلياً — وهو الخطأ الأخطر.")
    print(f"      • إنذار كاذب (FP): {fp} رابط شرعي حُجب = {fp/(fp+tn)*100:.1f}% من الروابط الشرعية.")
    print(f"        تكلفته إزعاج المستخدم، وإن كثر فقد ثقته في النظام وتجاهل تحذيراته.")
    print(f"      • هذه الأرقام ناتجة عن عتبة {threshold} المختارة من مجموعة التحقق")
    print(f"        بقاعدة: أعلى استدعاء بشرط ألا تتجاوز الإنذارات الكاذبة {MAX_FPR:.0%}.")

    # ---- رسم مصفوفة الالتباس -------------------------------------------
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legit", "Phishing"], yticklabels=["Legit", "Phishing"])
    plt.title(f"Confusion Matrix — {winner['family']} (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_fig("07_confusion_matrix.png")

    # ---- منحنى ROC -------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_te, m["_proba"])                   # نقاط المنحنى
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.4f}")  # المنحنى نفسه
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")        # خط التخمين العشوائي للمقارنة
    plt.title("ROC Curve (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_fig("08_roc_curve.png")

    # ---- فحص التسريب ------------------------------------------------------
    if m["accuracy"] > LEAKAGE_LIMIT:                            # رقم أعلى من اللازم = إشارة خطر
        print(f"\n   🚨 تحذير: الدقة {m['accuracy']:.4f} تتجاوز {LEAKAGE_LIMIT}.")
        print("      على هذه البيانات هذا يعني تسريباً غالباً. لا تقبل الرقم قبل تفسيره.")
    else:
        print(f"\n   ✅ الدقة {m['accuracy']:.4f} دون حد التسريب ({LEAKAGE_LIMIT}) — لا مؤشر تسريب.")

    # ---- أهم الخصائص -----------------------------------------------------
    if hasattr(winner["model"], "feature_importances_"):         # النماذج الشجرية فقط تملك هذه الخاصية
        imp = pd.Series(winner["model"].feature_importances_,
                        index=X_columns).sort_values(ascending=False)
        print("\n   ⭐ أهم 10 خصائص في قرار النموذج:")
        for name, val in imp.head(10).items():                   # طباعة أهم عشر خصائص
            print(f"      {name:28s} {val:.4f}")

        plt.figure(figsize=(8, 5))
        top = imp.head(15)                                       # نرسم أهم 15
        sns.barplot(x=top.values, y=top.index, hue=top.index,
                    palette="rocket", legend=False)
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        save_fig("09_feature_importance.png")

        imp.to_csv(os.path.join(FIG_DIR, "feature_importance.csv"),
                   header=["importance"], encoding="utf-8-sig")  # حفظ الترتيب كاملاً

    return m


# =========================================================
# 8. إثبات أثر التقسيم العشوائي (لتوثيقه في الرسالة)
# =========================================================
def random_split_demo(X, y, winner):
    """
    يدرّب نفس النموذج بتقسيم عشوائي بدل التقسيم حسب النطاق.

    الغرض ليس استخدام النتيجة، بل إثبات أنها متضخّمة: الفرق بين
    الرقمين هو حجم التسريب الذي تجنّبناه.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15,
                                              random_state=SEED, stratify=y)
    model = winner["model"].__class__(**winner["model"].get_params())  # نسخة جديدة بنفس المعاملات
    model.fit(X_tr, y_tr)                                        # تدريب على تقسيم عشوائي
    acc = accuracy_score(y_te, model.predict(X_te))              # قياس الدقة
    print(f"\n🔬 مقارنة منهجية:")
    print(f"   الدقة بتقسيم عشوائي        = {acc:.4f}  ← رقم متضخّم لا يُعتمد")
    return acc


# =========================================================
# 9. حفظ النموذج وأسماء الخصائص
# =========================================================
def save_model(winner, scaler, X_columns, test_metrics, threshold):
    """يحفظ النموذج ومعه كل ما يحتاجه التطبيق ليعطي نفس النتيجة."""
    bundle = {
        "model": winner["model"],                                # النموذج المدرَّب
        "scaler": scaler if winner["scaled"] else None,          # المطبّع (فقط للنماذج الخطية)
        "threshold": threshold,                                  # عتبة القرار المختارة من التحقق
        "feature_names": list(X_columns),                        # ترتيب الخصائص — حاسم للتوافق
        "model_name": winner["family"],                          # اسم النموذج للتوثيق
        "params": winner["params"],                              # معاملاته
        "test_metrics": {k: float(v) for k, v in test_metrics.items()
                         if not k.startswith("_")},              # نتائجه على الاختبار
    }
    joblib.dump(bundle, MODEL_FILE)                              # الحفظ في ملف واحد
    print(f"\n💾 حُفظ النموذج: {MODEL_FILE}")

    with open(FEATURES_FILE, "w", encoding="utf-8") as fh:       # ملف منفصل لأسماء الخصائص
        json.dump(list(X_columns), fh, ensure_ascii=False, indent=2)
    print(f"💾 حُفظت أسماء الخصائص: {FEATURES_FILE} ({len(X_columns)} خاصية)")
    print("   ➜ التطبيق في المرحلة 3 يجب أن يبني المتجه بهذا الترتيب بالضبط.")


# =========================================================
# دالة مساعدة لحفظ الرسومات
# =========================================================
def save_fig(filename):
    """تحفظ الرسمة الحالية بدقة 300 ثم تغلقها."""
    path = os.path.join(FIG_DIR, filename)                       # المسار الكامل
    plt.tight_layout()                                           # ترتيب العناصر تلقائياً
    plt.savefig(path, dpi=300, bbox_inches="tight")              # الحفظ بدقة عالية
    plt.close()                                                  # تحرير الذاكرة
    print(f"   🖼️  حُفظت: {path}")


# =========================================================
# التشغيل الرئيسي
# =========================================================
def main():
    print("=" * 60)
    print("المرحلة 2 — التدريب والمقارنة")
    print("=" * 60)
    os.makedirs(FIG_DIR, exist_ok=True)                          # إنشاء مجلد الرسومات إن لم يوجد

    X, y, groups = load_data()                                   # 1. تحميل البيانات

    print("\n✂️ التقسيم حسب النطاق (GroupShuffleSplit):")
    parts = split_by_domain(X, y, groups)                        # 2. التقسيم الثلاثي

    # 3. التطبيع: يُضبط على بيانات التدريب وحدها حتى لا تتسرب إحصاءات التحقق والاختبار
    scaler = StandardScaler().fit(parts["train"][0])
    print("   📏 ضُبط المطبّع على بيانات التدريب فقط (للنماذج الخطية وحدها)")

    results, best = train_all(parts, scaler)                     # 4. تدريب المرشحين
    comparison_table(best)                                       # 5. جدول المقارنة

    winner = max(best.values(), key=lambda r: r["f1"])           # 6. الفائز = أعلى F1 على التحقق
    print(f"\n🏆 النموذج المختار: {winner['family']} ({winner['params']})")

    threshold = choose_threshold(winner, parts, scaler)          # 7. اختيار العتبة من التحقق
    test_metrics = final_test(winner, parts, scaler, X.columns, threshold)  # 8. الاختبار النهائي
    random_acc = random_split_demo(X, y, winner)                 # 9. مقارنة التقسيم العشوائي
    print(f"   الدقة بالتقسيم حسب النطاق  = {test_metrics['accuracy']:.4f}  ← الرقم المعتمد")
    print(f"   حجم التضخّم                = {(random_acc - test_metrics['accuracy'])*100:.1f} نقطة مئوية")

    save_model(winner, scaler, X.columns, test_metrics, threshold)  # 10. الحفظ

    print("\n➜ المرحلة 2 اكتملت. التالي: المرحلة 3 (واجهة FastAPI).")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ ملف مفقود: {e}. شغّل 00_build_dataset.py أولاً.")
    except Exception as e:
        print(f"❌ حدث خطأ غير متوقع: {e}")
