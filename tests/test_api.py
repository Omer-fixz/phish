# =========================================================
# اختبار الـ API — يعمل بدون تشغيل خادم منفصل
# =========================================================
# يستخدم TestClient من FastAPI الذي يستدعي التطبيق مباشرة في الذاكرة.
# =========================================================

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys, pathlib
# نضيف جذر المشروع إلى مسار البحث حتى نستطيع استيراد src من أي مجلد
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


from fastapi.testclient import TestClient
from src.api import main
from src.api.main import app, THRESHOLD


def run_api_checks():
    """يجري مجموعة اختبارات API ويُرجع رمز الخروج المناسب."""
    client = TestClient(app)          # عميل وهمي يتحدث مع التطبيق مباشرة
    passed = failed = 0

    def check(label, ok):
        """يسجّل نتيجة اختبار واحد ويطبعها."""
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  [ نجح ] {label}")
        else:
            failed += 1
            print(f"  [ فشل ] {label}")

    print("\n=== 1. مسار الصحة ===")
    r = client.get("/health")
    check("يرد بـ 200", r.status_code == 200)
    check("يذكر اسم النموذج", "model" in r.json())
    print(f"        {r.json()}")

    print("\n=== 2. روابط شرعية ===")
    for u in ["google.com", "www.google.com", "https://www.bbc.co.uk/news/world",
              "github.com/python/cpython", "www.uofk.edu"]:
        r = client.post("/predict", json={"url": u})
        j = r.json()
        check(f"{u} -> {j['prediction']} ({j['phishing_probability']})",
              j["prediction"] == "legitimate")

    print("\n=== 3. روابط تصيد ===")
    for u in ["paypal.secure-login.tk/verify/account.php",
              "evil-site.xyz/paypal/webscr?cmd=login&update=1",
              "192.168.0.44/apple/id/signin.html",
              "xn--pypal-4ve.com/webscr?cmd=login",
              "bit.ly/3xK9pQ"]:
        r = client.post("/predict", json={"url": u})
        j = r.json()
        check(f"{u[:45]} -> {j['prediction']} ({j['phishing_probability']})",
              j["prediction"] == "phishing")

    print("\n=== 4. شكل الرد ===")
    r = client.post("/predict", json={"url": "paypal.secure-login.tk/verify"})
    j = r.json()
    check("يحتوي كل الحقول المطلوبة",
          all(k in j for k in ["url", "prediction", "label", "confidence",
                               "phishing_probability", "threshold", "top_features"]))
    check("عدد الخصائص المفسِّرة = 3", len(j["top_features"]) == 3)
    check("كل خاصية مفسِّرة لها اسم وقيمة ومساهمة",
          all(all(k in f for k in ["name", "value", "contribution", "direction"])
              for f in j["top_features"]))
    check("الثقة بين 0 و 1", 0 <= j["confidence"] <= 1)
    check("العتبة معلنة في الرد", j["threshold"] == THRESHOLD)
    print("        أهم الخصائص:")
    for f in j["top_features"]:
        print(f"          {f['name']} = {f['value']}  ({f['direction']}، {f['contribution']})")

    print("\n=== 5. رفض المدخلات غير الصالحة ===")
    bad_inputs = {
        "رابط فارغ": "",
        "مسافات فقط": "   ",
        "بلا نقطة": "notaurl",
        "فيه مسافة": "http://a b.com",
        "javascript:": "javascript:alert(1)",
        "data:": "data:text/html,<script>alert(1)</script>",
        "أطول من الحد": "http://a.com/" + "x" * 3000,
    }
    for label, u in bad_inputs.items():
        r = client.post("/predict", json={"url": u})
        check(f"{label} يُرفض (كود {r.status_code})", r.status_code == 422)

    r = client.post("/predict", json={})                     # حقل url مفقود تماماً
    check(f"طلب بلا حقل url يُرفض (كود {r.status_code})", r.status_code == 422)

    print("\n=== 6. الاتساق مع سكربت التدريب ===")
    # نفس الرابط يجب أن يعطي نفس المتجه في الـAPI وفي الاستخراج المباشر
    import json as _json
    from src.feature_extractor import extract_features

    from src.paths import FEATURES_JSON
    names = _json.load(open(FEATURES_JSON, encoding="utf-8"))
    u = "paypal.secure-login.tk/verify?id=99"
    direct = [float(extract_features(u)[n]) for n in names]           # الاستخراج المباشر
    api_val = {f["name"]: f["value"] for f in
               client.post("/predict", json={"url": u}).json()["top_features"]}
    check("قيم الخصائص في الرد تطابق الاستخراج المباشر",
          all(abs(direct[names.index(k)] - v) < 1e-9 for k, v in api_val.items()))

    print("\n=== 7. صفحة الواجهة (المرحلة 4) ===")
    r = client.get("/")
    check("الصفحة تُقدَّم من نفس الخادم (200)", r.status_code == 200)
    html = r.text
    check("الصفحة عربية باتجاه rtl", 'dir="rtl"' in html)
    check("فيها حقل الإدخال وزر الفحص", 'id="urlInput"' in html and 'id="checkBtn"' in html)
    check("تستخدم textContent لعرض الرابط (حماية من XSS)",
          'shownUrl").textContent' in html)
    check("لا تحوّل الرابط المفحوص إلى وسم <a> قابل للنقر",
          "createElement(\"a\")" not in html and "<a href" not in html)

    # رابط يحوي وسم HTML: يجب أن يعود كما هو نصاً في الرد، لا أن يُنفَّذ
    xss = "evil.com/<script>alert(1)</script>"      # بلا مسافات حتى يتجاوز فحص المسافة ويصل للنموذج
    r = client.post("/predict", json={"url": xss})
    check(f"رابط فيه وسم HTML يُعالَج كنص (كود {r.status_code})",
          r.status_code == 200 and r.json()["url"] == xss)

    print("\n=== 8. التأمين (المرحلة 5) ===")

    r = client.post("/predict", json={"url": "google.com"})
    h = r.headers
    check("ترويسة nosniff موجودة", h.get("X-Content-Type-Options") == "nosniff")
    check("منع التأطير (X-Frame-Options)", h.get("X-Frame-Options") == "DENY")
    check("منع تسريب الرابط عبر Referer", h.get("Referrer-Policy") == "no-referrer")
    check("سياسة محتوى (CSP) معرَّفة", "default-src 'self'" in h.get("Content-Security-Policy", ""))
    check("CSP تمنع التأطير أيضاً", "frame-ancestors 'none'" in h.get("Content-Security-Policy", ""))
    check("CORS لا يسمح للجميع (*)", "*" not in main.ALLOWED_ORIGINS)

    # --- اختبار حد الطلبات ---------------------------------------------------
    main._hits.clear()                                 # تصفير العدّاد ليبدأ الاختبار من الصفر
    codes = [client.post("/predict", json={"url": "google.com"}).status_code
             for _ in range(main.RATE_LIMIT + 5)]      # نتجاوز الحد عمداً
    check(f"أول {main.RATE_LIMIT} طلباً مسموحة",
          codes[:main.RATE_LIMIT] == [200] * main.RATE_LIMIT)
    check("ما بعد الحد يُرفض بكود 429", all(c == 429 for c in codes[main.RATE_LIMIT:]))

    blocked = client.post("/predict", json={"url": "google.com"})
    check("رد الحظر فيه ترويسة Retry-After", "Retry-After" in blocked.headers)
    check("رسالة الحظر عربية مفهومة", "طلبات كثيرة" in blocked.json()["detail"])

    main._hits.clear()                                 # تصفير العدّاد بعد الاختبار
    check("مسار /health لا يخضع للحد", client.get("/health").status_code == 200)

    # --- التأكد أن الرابط الكامل لا يُسجَّل ------------------------------------
    import io as _io, logging as _logging, re as _re
    buf = _io.StringIO()                               # مخزن مؤقت نلتقط فيه السجلات
    handler = _logging.StreamHandler(buf)
    main.log.addHandler(handler)
    secret = "bank.example.com/account?session_token=SUPERSECRET123"
    client.post("/predict", json={"url": secret})
    main.log.removeHandler(handler)
    logged = buf.getvalue()
    check("الرابط الكامل لا يظهر في السجل", secret not in logged)
    check("الرمز السري لا يظهر في السجل", "SUPERSECRET123" not in logged)
    check("السجل يحوي بصمة قصيرة للتتبع", bool(_re.search(r"بصمة=[0-9a-f]{10}", logged)))

    print("\n" + "=" * 55)
    print(f"النتيجة: نجح {passed} | فشل {failed}")
    print("=" * 55)
    return 1 if failed else 0


def test_api_smoke():
    assert run_api_checks() == 0


if __name__ == "__main__":
    raise SystemExit(run_api_checks())
