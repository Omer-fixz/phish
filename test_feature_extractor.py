"""
test_feature_extractor.py
=========================
اختبارات وحدة (Unit Tests) للتأكد من صحة استخراج كل خاصية.

طريقة التشغيل:
    python test_feature_extractor.py

هذه الاختبارات ليست رفاهية. الغرض منها:
    1. إثبات أن كل خاصية تعمل كما هو موصوف في الرسالة.
    2. الحماية من كسر الكود عند أي تعديل مستقبلي.
    3. مادة جاهزة لفصل "اختبار النظام" في الرسالة (البند 4.4).
"""

import sys

from feature_extractor import (
    FEATURE_NAMES,
    extract_features,
    extract_features_batch,
    features_to_vector,
)

_passed = 0
_failed = 0


def check(label, actual, expected):
    global _passed, _failed
    if actual == expected:
        _passed += 1
        print(f"  [ نجح  ] {label}")
    else:
        _failed += 1
        print(f"  [ فشل  ] {label}  |  المتوقع={expected}  الناتج={actual}")


def approx(label, actual, expected, tol=0.01):
    global _passed, _failed
    if abs(actual - expected) <= tol:
        _passed += 1
        print(f"  [ نجح  ] {label}")
    else:
        _failed += 1
        print(f"  [ فشل  ] {label}  |  المتوقع≈{expected}  الناتج={actual}")


# ===========================================================================
print("\n=== 1. سلامة البنية العامة ===")
# ===========================================================================
f = extract_features("https://www.example.com/path?a=1")
check("عدد الخصائص = 37", len(f), 37)
check("لا تكرار في أسماء الخصائص", len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
check("مفاتيح المخرجات مطابقة لـ FEATURE_NAMES",
      list(f.keys()), FEATURE_NAMES)
check("كل القيم أرقام",
      all(isinstance(v, (int, float)) for v in f.values()), True)
check("طول المتجه = 37", len(features_to_vector("http://a.com")), 37)


# ===========================================================================
print("\n=== 2. معالجة المدخلات المشوّهة (لا يجب أن ينهار) ===")
# ===========================================================================
edge_cases = ["", "   ", "not a url", "http://", "///", "@@@@",
              "http://[::1]/x", "ftp://files.example.com/a.txt",
              "http://a.com:99999/", '"http://quoted.com"']
crashed = []
for case in edge_cases:
    try:
        r = extract_features(case)
        if len(r) != 37:
            crashed.append(case)
    except Exception as e:  # noqa: BLE001
        crashed.append(f"{case} -> {e}")
check("كل الحالات الحدّية تُعالَج بدون انهيار", crashed, [])

check("None لا يسبب خطأ", len(extract_features(None)), 37)


# ===========================================================================
print("\n=== 3. إضافة البروتوكول تلقائياً ===")
# ===========================================================================
# رابط بدون http:// يجب أن يُفهم مضيفه بشكل صحيح
a = extract_features("www.google.com/search")
check("الرابط بدون بروتوكول: يُتعرّف على المضيف",
      a["hostname_length"] > 0, True)
check("الرابط بدون بروتوكول: عمق المسار = 1", a["path_depth"], 1)


# ===========================================================================
print("\n=== 4. عنوان IP بدلاً من اسم نطاق ===")
# ===========================================================================
check("IPv4 يُكتشف",
      extract_features("http://192.168.1.1/login")["has_ip_address"], 1)
check("IPv6 يُكتشف",
      extract_features("http://[2001:db8::1]/a")["has_ip_address"], 1)
check("نطاق عادي لا يُعد IP",
      extract_features("http://google.com")["has_ip_address"], 0)
check("نطاق فيه أرقام لا يُعد IP",
      extract_features("http://web3.example.com")["has_ip_address"], 0)


# ===========================================================================
print("\n=== 5. الرمز @ (يتجاوز المتصفح كل ما قبله) ===")
# ===========================================================================
# http://google.com@evil.com يذهب فعلياً إلى evil.com
tricky = extract_features("http://google.com@evil.com/login")
check("الرمز @ يُكتشف", tricky["has_at_symbol"], 1)
check("رابط عادي: لا يوجد @",
      extract_features("http://google.com")["has_at_symbol"], 0)


# ===========================================================================
print("\n=== 6. خدمات اختصار الروابط ===")
# ===========================================================================
check("bit.ly يُكتشف",
      extract_features("https://bit.ly/3xK9pQ")["is_shortened_url"], 1)
check("tinyurl يُكتشف",
      extract_features("http://tinyurl.com/abcd")["is_shortened_url"], 1)
check("موقع عادي لا يُعد مختصراً",
      extract_features("https://www.bbc.com/news")["is_shortened_url"], 0)


# ===========================================================================
print("\n=== 7. Punycode (هجوم التشابه البصري) ===")
# ===========================================================================
check("xn-- يُكتشف",
      extract_features("http://xn--pypal-4ve.com")["is_punycode"], 1)
check("نطاق لاتيني عادي: لا punycode",
      extract_features("http://paypal.com")["is_punycode"], 0)


# ===========================================================================
print("\n=== 8. أقوى خاصية: العلامة التجارية في المكان الخطأ ===")
# ===========================================================================
check("paypal.com الشرعي -> 0",
      extract_features("https://www.paypal.com/signin")["brand_in_wrong_place"], 0)
check("paypal في النطاق الفرعي -> 1",
      extract_features("http://paypal.secure-login.tk/")["brand_in_wrong_place"], 1)
check("paypal في المسار -> 1",
      extract_features("http://evil-site.xyz/paypal/verify")["brand_in_wrong_place"], 1)
check("apple في نطاق فرعي مركّب -> 1",
      extract_features("http://apple.id.verify.ru/")["brand_in_wrong_place"], 1)
check("موقع بلا علامات تجارية -> 0",
      extract_features("https://www.uofk.edu/about")["brand_in_wrong_place"], 0)


# ===========================================================================
print("\n=== 9. النطاقات العليا المشبوهة ===")
# ===========================================================================
check(".tk مشبوه", extract_features("http://x.tk")["has_suspicious_tld"], 1)
check(".xyz مشبوه", extract_features("http://x.xyz")["has_suspicious_tld"], 1)
check(".com غير مشبوه", extract_features("http://x.com")["has_suspicious_tld"], 0)
check(".edu غير مشبوه", extract_features("http://x.edu")["has_suspicious_tld"], 0)
check("co.uk يُقرأ صحيحاً (uk وليس co)",
      extract_features("http://bbc.co.uk")["has_suspicious_tld"], 0)


# ===========================================================================
print("\n=== 10. كلمة https داخل اسم النطاق (خدعة بصرية) ===")
# ===========================================================================
check("https-paypal.com يُكتشف",
      extract_features("http://https-paypal-secure.com")["https_token_in_domain"], 1)
check("نطاق نظيف -> 0",
      extract_features("https://paypal.com")["https_token_in_domain"], 0)


# ===========================================================================
print("\n=== 11. عدّ النطاقات الفرعية (مع تجاهل www) ===")
# ===========================================================================
check("www.google.com -> 0 (www لا تُحسب)",
      extract_features("https://www.google.com")["subdomain_count"], 0)
check("mail.google.com -> 1",
      extract_features("https://mail.google.com")["subdomain_count"], 1)
check("a.b.c.example.com -> 3",
      extract_features("http://a.b.c.example.com")["subdomain_count"], 3)


# ===========================================================================
print("\n=== 12. الكلمات المشبوهة ===")
# ===========================================================================
susp = extract_features("http://x.tk/login/verify/secure/account/update")
check("خمس كلمات مشبوهة على الأقل",
      susp["suspicious_keyword_count"] >= 5, True)
check("موقع نظيف: صفر كلمات مشبوهة",
      extract_features("https://www.wikipedia.org/wiki/Sudan")["suspicious_keyword_count"], 0)
# الترميز السداسي لا يخفي الكلمة عن الكاشف
check("الكلمة المخفية بترميز %6C%6F%67%69%6E تُكتشف",
      extract_features("http://x.tk/%6C%6F%67%69%6E")["suspicious_keyword_count"] >= 1, True)
check("الترميز السداسي يُكتشف",
      extract_features("http://x.tk/%6C%6F%67")["has_hex_encoding"], 1)


# ===========================================================================
print("\n=== 13. المنافذ غير القياسية ===")
# ===========================================================================
check("المنفذ 8080 غير قياسي",
      extract_features("http://x.com:8080/a")["has_non_standard_port"], 1)
check("المنفذ 443 قياسي",
      extract_features("https://x.com:443/a")["has_non_standard_port"], 0)
check("بدون منفذ -> 0",
      extract_features("https://x.com/a")["has_non_standard_port"], 0)


# ===========================================================================
print("\n=== 14. النسب والإنتروبيا ===")
# ===========================================================================
r = extract_features("http://abc.com")
approx("مجموع النسب الثلاث = 1",
       r["digit_ratio"] + r["letter_ratio"] + r["special_char_ratio"], 1.0)

plain = extract_features("http://google.com")["entropy_domain"]
random_like = extract_features("http://x7f2kq9zm4vb.com")["entropy_domain"]
check("النطاق العشوائي إنتروبيته أعلى من النطاق المفهوم",
      random_like > plain, True)


# ===========================================================================
print("\n=== 15. الاتساق: نفس الرابط يعطي نفس النتيجة دائماً ===")
# ===========================================================================
u = "http://paypal.secure-login.tk/verify?id=99"
check("استدعاءان متتاليان يعطيان نفس النتيجة",
      extract_features(u), extract_features(u))
check("features_to_vector مطابق لـ extract_features",
      features_to_vector(u),
      [extract_features(u)[n] for n in FEATURE_NAMES])


# ===========================================================================
print("\n=== 16. المعالجة الجماعية (Batch) ===")
# ===========================================================================
urls = ["https://google.com", "http://paypal.login.tk/x", "bit.ly/abc"]
df = extract_features_batch(urls)
check("عدد الصفوف صحيح", len(df), 3)
check("ترتيب الأعمدة مطابق لـ FEATURE_NAMES", list(df.columns), FEATURE_NAMES)
check("لا توجد قيم مفقودة", int(df.isnull().sum().sum()), 0)
check("صف الدفعة مطابق للاستخراج الفردي",
      df.iloc[1].tolist(), features_to_vector(urls[1]))


# ===========================================================================
print("\n=== 17. اختبار تمييزي: تصيد مقابل شرعي ===")
# ===========================================================================
legit = ["https://www.google.com",
         "https://www.uofk.edu/faculties",
         "https://github.com/python/cpython",
         "https://www.bbc.co.uk/news"]
phish = ["http://paypal.secure-verify.tk/login/account.php",
         "http://192.168.0.44/apple/id/signin.html",
         "http://xn--pypal-4ve.com/webscr?cmd=login",
         "http://microsoft.office365.verify-account.xyz/update"]


def risk(u):
    """مؤشر خطورة تقريبي للتحقق من قدرة الخصائص على التمييز."""
    x = extract_features(u)
    return (x["has_ip_address"] + x["is_punycode"] +
            x["brand_in_wrong_place"] + x["has_suspicious_tld"] +
            min(x["suspicious_keyword_count"], 3))


avg_legit = sum(risk(u) for u in legit) / len(legit)
avg_phish = sum(risk(u) for u in phish) / len(phish)
print(f"  متوسط مؤشر الخطورة - شرعي: {avg_legit:.2f} | تصيد: {avg_phish:.2f}")
check("الروابط الخبيثة تسجل خطورة أعلى بوضوح",
      avg_phish > avg_legit + 2, True)


# ===========================================================================
print("\n" + "=" * 60)
print(f"النتيجة النهائية:  نجح {_passed}  |  فشل {_failed}")
print("=" * 60)
sys.exit(1 if _failed else 0)
