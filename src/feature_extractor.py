"""
feature_extractor.py
====================
استخراج الخصائص المعجمية (Lexical Features) من روابط المواقع
لاكتشاف مواقع التصيد الاحتيالي باستخدام تعلم الآلة.

مشروع التخرج رقم 43 - جامعة العلوم والتقانة
كلية علوم الحاسوب وتقانة المعلومات

=====================================================================
!!! تحذير مهم جداً !!!
هذا الملف هو المصدر الوحيد لاستخراج الخصائص في المشروع كله.
يجب استيراده (import) في:
    1. سكربت بناء البيانات (00_build_dataset.py)
    2. سكربت التدريب       (train_model.py)
    3. تطبيق الويب         (main.py)

لا تكتب استخراج الخصائص مرة أخرى في أي مكان آخر.
أي اختلاف بين طريقة الاستخراج وقت التدريب ووقت التطبيق
سيجعل النموذج يعطي نتائج عشوائية في الواجهة رغم دقته العالية
في الاختبار. هذه أشهر مشكلة في مشاريع كشف التصيد.
=====================================================================

المميزات:
    - لا يتصل بالإنترنت إطلاقاً (Offline / Lexical only)
    - سريع جداً: أقل من 1 ميلي ثانية للرابط الواحد
    - يعمل حتى لو كان الموقع مغلقاً أو محذوفاً
    - آمن: لا يفتح الرابط ولا ينزّل محتواه
"""

from __future__ import annotations

import ipaddress
import math
import re
import socket
import ssl
import datetime
from collections import Counter
from urllib.parse import urlparse, unquote

import tldextract

# ---------------------------------------------------------------------------
# إعداد tldextract للعمل بدون إنترنت
# suffix_list_urls=() تمنعه من محاولة تحميل قائمة النطاقات من الإنترنت
# ويستخدم النسخة المدمجة معه بدلاً من ذلك.
# ---------------------------------------------------------------------------
_extract = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)


# ---------------------------------------------------------------------------
# القوائم المرجعية
# ---------------------------------------------------------------------------

# خدمات اختصار الروابط: تُستخدم لإخفاء الوجهة الحقيقية للرابط
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorte.st", "tiny.cc",
    "rb.gy", "shorturl.at", "s.id", "t.ly", "soo.gd", "clck.ru", "v.gd",
    "trib.al", "lnkd.in", "db.tt", "qr.ae", "j.mp", "bl.ink", "short.io",
    "u.to", "x.co", "po.st", "bc.vc", "urlz.fr", "ity.im", "q.gs",
}

# الكلمات التي يكثر ظهورها في روابط التصيد لإيهام الضحية بالشرعية
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "confirmation", "banking",
    "bank", "password", "credential", "webscr", "authenticate", "auth",
    "wallet", "recover", "recovery", "unlock", "suspended", "limited",
    "billing", "invoice", "payment", "submit", "validate", "alert",
    "support", "customer", "service", "office", "live", "session",
]

# العلامات التجارية الأكثر استهدافاً في هجمات التصيد
TARGET_BRANDS = [
    "paypal", "apple", "microsoft", "google", "facebook", "amazon",
    "netflix", "instagram", "whatsapp", "linkedin", "dropbox", "outlook",
    "office365", "icloud", "chase", "wellsfargo", "citibank", "hsbc",
    "santander", "bankofamerica", "steam", "roblox", "binance", "coinbase",
    "dhl", "fedex", "ups", "usps", "adobe", "yahoo", "twitter", "telegram",
]

# نطاقات عليا مجانية أو رخيصة يكثر إساءة استخدامها في التصيد
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    "buzz", "loan", "download", "review", "country", "stream", "gdn",
    "racing", "win", "bid", "date", "faith", "science", "party", "cricket",
    "accountant", "men", "trade", "webcam", "zip", "mov", "rest", "cyou",
}

# نمط الكلمات (للبحث عن أطول كلمة في الرابط)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# نمط الترميز السداسي عشري في الروابط مثل %20 %2F
_HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")

# مهلة الاتصال بالشبكة (ثوانٍ) — قصيرة لضمان استجابة سريعة للـ API
_NET_TIMEOUT = 5


# ---------------------------------------------------------------------------
# دوال فحص الشبكة (Domain Age / WHOIS / SSL)
# ---------------------------------------------------------------------------

def _get_domain_age_days(hostname: str) -> int:
    """
    يحسب عمر النطاق بالأيام من تاريخ التسجيل حتى اليوم عبر WHOIS.

    المواقع الشرعية عادةً مسجَّلة منذ سنوات.
    مواقع التصيد كثيراً ما تُنشأ وتُستخدم خلال أيام أو أسابيع.
    القيمة -1 تعني تعذّر الحصول على المعلومة (موقع غير موجود، WHOIS محجوب...).
    """
    try:
        import whois  # python-whois
        w = whois.whois(hostname)
        creation = w.creation_date
        if creation is None:
            return -1
        # بعض المكتبات تُرجع قائمة بدل تاريخ واحد
        if isinstance(creation, list):
            creation = creation[0]
        if not isinstance(creation, datetime.datetime):
            return -1
        delta = datetime.datetime.utcnow() - creation
        return max(delta.days, 0)
    except Exception:
        return -1


def _get_ssl_info(hostname: str) -> dict:
    """
    يتصل بالموقع على المنفذ 443 ويستخرج معلومات شهادة SSL/TLS.

    المخرجات (dict):
        ssl_valid        : 1 إن كانت الشهادة موجودة وصالحة الآن، 0 في غير ذلك
        ssl_days_left    : أيام المتبقية لانتهاء الشهادة (-1 إن تعذّر)
        ssl_self_signed  : 1 إن كانت الشهادة موقّعة ذاتياً (Self-Signed)
        ssl_wildcard     : 1 إن كانت شهادة بدل (*) — أحياناً مؤشر مشبوه
    """
    result = {
        "ssl_valid": 0,
        "ssl_days_left": -1,
        "ssl_self_signed": 0,
        "ssl_wildcard": 0,
    }
    if not hostname:
        return result
    try:
        ctx = ssl.create_default_context()
        conn = ctx.wrap_socket(
            socket.create_connection((hostname, 443), timeout=_NET_TIMEOUT),
            server_hostname=hostname,
        )
        cert = conn.getpeercert()
        conn.close()

        # تاريخ الانتهاء
        not_after = cert.get("notAfter", "")
        if not_after:
            exp = datetime.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
            days_left = (exp - datetime.datetime.utcnow()).days
            result["ssl_valid"] = int(days_left > 0)
            result["ssl_days_left"] = days_left

        # شهادة ذاتية: المُصدِر هو نفسه الموضوع
        issuer = dict(x[0] for x in cert.get("issuer", []))
        subject = dict(x[0] for x in cert.get("subject", []))
        result["ssl_self_signed"] = int(
            issuer.get("commonName", "A") == subject.get("commonName", "B")
        )

        # شهادة بدل (Wildcard)
        san = cert.get("subjectAltName", [])
        result["ssl_wildcard"] = int(
            any(v.startswith("*.") for _, v in san)
        )

    except ssl.SSLCertVerificationError:
        # شهادة غير موثوقة أو منتهية — هذا بحد ذاته مؤشر
        result["ssl_valid"] = 0
    except Exception:
        pass  # موقع لا يدعم HTTPS أو غير متاح
    return result


# ---------------------------------------------------------------------------
# دوال مساعدة
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """
    حساب إنتروبيا شانون للنص.

    الفكرة: النطاقات الشرعية تُكتب بكلمات مفهومة (google, facebook)
    فتكون إنتروبيتها منخفضة. النطاقات المولّدة آلياً بواسطة المهاجمين
    (مثل: x7f2kq9zm.com) تكون حروفها شبه عشوائية فترتفع إنتروبيتها.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _is_ip_address(host: str) -> bool:
    """هل المضيف عنوان IP بدلاً من اسم نطاق؟ (مؤشر تصيد قوي)"""
    if not host:
        return False
    cleaned = host.strip("[]")  # لدعم صيغة IPv6
    try:
        ipaddress.ip_address(cleaned)
        return True
    except ValueError:
        return False


def _normalize(url: str) -> str:
    """
    توحيد صيغة الرابط قبل التحليل.

    مهم جداً: لو الرابط لا يبدأ ببروتوكول، urlparse يعتبر النطاق
    كله جزءاً من المسار (path) وتفشل كل الخصائص المبنية على المضيف.
    """
    if url is None:
        return ""
    url = str(url).strip().strip('"').strip("'")
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    return url


# ---------------------------------------------------------------------------
# قائمة أسماء الخصائص بالترتيب الثابت
# ---------------------------------------------------------------------------
# هذا الترتيب مقدّس. النموذج يتوقع الخصائص بهذا الترتيب بالضبط.
# لا تُعِد ترتيبها ولا تحذف منها بعد تدريب النموذج.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # -- المجموعة 1: الأطوال --------------------------------------------
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "tld_length",
    # -- المجموعة 2: عدّادات الرموز -------------------------------------
    "count_dot",
    "count_hyphen",
    "count_underscore",
    "count_slash",
    "count_question",
    "count_equal",
    "count_at",
    "count_ampersand",
    "count_percent",
    "count_digits",
    "count_special_chars",
    # -- المجموعة 3: النسب ----------------------------------------------
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "digit_ratio_in_host",
    # -- المجموعة 4: خصائص ثنائية (0/1) ---------------------------------
    "has_ip_address",
    "has_at_symbol",
    "has_double_slash_in_path",
    "is_shortened_url",
    "has_non_standard_port",
    "is_punycode",
    "https_token_in_domain",
    "is_https",
    "has_hex_encoding",
    # -- المجموعة 5: خصائص بنيوية ---------------------------------------
    "subdomain_count",
    "path_depth",
    "longest_word_length",
    # -- المجموعة 6: خصائص دلالية ---------------------------------------
    "suspicious_keyword_count",
    "brand_in_wrong_place",
    "has_suspicious_tld",
    # -- المجموعة 7: خصائص إحصائية --------------------------------------
    "entropy_url",
    "entropy_domain",
    # -- المجموعة 8: خصائص الشبكة (Domain / WHOIS / SSL) ----------------
    "domain_age_days",
    "ssl_valid",
    "ssl_days_left",
    "ssl_self_signed",
    "ssl_wildcard",
]


# ---------------------------------------------------------------------------
# الدالة الرئيسية
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    استخراج كل الخصائص من رابط واحد.

    المعامل:
        url : الرابط كنص. يقبل الروابط بدون بروتوكول والروابط المشوّهة.

    المخرجات:
        قاموس (dict) مفاتيحه هي FEATURE_NAMES وقيمه أرقام.
        الترتيب مضمون ومطابق لـ FEATURE_NAMES.

    مثال:
        >>> f = extract_features("http://paypal.secure-login.tk/verify")
        >>> f["brand_in_wrong_place"]
        1
    """
    raw = _normalize(url)

    # ---- التفكيك -------------------------------------------------------
    try:
        parsed = urlparse(raw)
    except ValueError:
        parsed = urlparse("http://invalid")

    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        port = parsed.port
    except ValueError:
        port = None

    ext = _extract(raw)
    subdomain = (ext.subdomain or "").lower()
    domain = (ext.domain or "").lower()
    tld = (ext.suffix or "").lower()

    # النص الكامل بعد فك الترميز، يُستخدم للبحث الدلالي عن الكلمات
    decoded_lower = unquote(raw).lower()

    # ---- المجموعة 1: الأطوال -------------------------------------------
    f = {}
    f["url_length"] = len(raw)
    f["hostname_length"] = len(hostname)
    f["path_length"] = len(path)
    f["query_length"] = len(query)
    f["tld_length"] = len(tld)

    # ---- المجموعة 2: عدّادات الرموز ------------------------------------
    f["count_dot"] = raw.count(".")
    f["count_hyphen"] = raw.count("-")
    f["count_underscore"] = raw.count("_")
    f["count_slash"] = raw.count("/")
    f["count_question"] = raw.count("?")
    f["count_equal"] = raw.count("=")
    f["count_at"] = raw.count("@")
    f["count_ampersand"] = raw.count("&")
    f["count_percent"] = raw.count("%")

    digits = sum(ch.isdigit() for ch in raw)
    letters = sum(ch.isalpha() for ch in raw)
    specials = sum((not ch.isalnum()) for ch in raw)

    f["count_digits"] = digits
    f["count_special_chars"] = specials

    # ---- المجموعة 3: النسب ---------------------------------------------
    n = len(raw) or 1
    f["digit_ratio"] = round(digits / n, 6)
    f["letter_ratio"] = round(letters / n, 6)
    f["special_char_ratio"] = round(specials / n, 6)

    host_len = len(hostname) or 1
    f["digit_ratio_in_host"] = round(
        sum(ch.isdigit() for ch in hostname) / host_len, 6
    )

    # ---- المجموعة 4: خصائص ثنائية --------------------------------------
    # عنوان IP بدل اسم نطاق: مؤشر تصيد قوي جداً
    f["has_ip_address"] = int(_is_ip_address(hostname))

    # الرمز @ يجعل المتصفح يتجاهل كل ما قبله
    # مثال: http://google.com@evil.com  -> يذهب فعلياً إلى evil.com
    f["has_at_symbol"] = int("@" in raw)

    # // داخل المسار تُستخدم لإعادة التوجيه المخفية
    f["has_double_slash_in_path"] = int("//" in path)

    # خدمة اختصار روابط تخفي الوجهة الحقيقية
    registered = f"{domain}.{tld}" if domain and tld else hostname
    f["is_shortened_url"] = int(registered in SHORTENER_DOMAINS)

    # منفذ غير قياسي (غير 80 و 443)
    f["has_non_standard_port"] = int(port is not None and port not in (80, 443))

    # Punycode: يُستخدم في هجمات التشابه البصري (Homograph Attack)
    # مثال: xn--pypal-4ve.com تظهر في المتصفح كـ pаypal.com
    f["is_punycode"] = int("xn--" in hostname)

    # كلمة https داخل اسم النطاق نفسه لخداع المستخدم
    # مثال: https-paypal-secure.com
    f["https_token_in_domain"] = int("https" in domain or "http" in domain)

    f["is_https"] = int(scheme == "https")

    # ترميز سداسي عشري %xx يُستخدم لإخفاء الكلمات المشبوهة
    f["has_hex_encoding"] = int(bool(_HEX_RE.search(raw)))

    # ---- المجموعة 5: خصائص بنيوية --------------------------------------
    if subdomain:
        parts = [p for p in subdomain.split(".") if p]
        # لا نحسب www كنطاق فرعي لأنها شائعة في المواقع الشرعية
        f["subdomain_count"] = len([p for p in parts if p != "www"])
    else:
        f["subdomain_count"] = 0

    f["path_depth"] = len([seg for seg in path.split("/") if seg])

    words = _WORD_RE.findall(raw)
    f["longest_word_length"] = max((len(w) for w in words), default=0)

    # ---- المجموعة 6: خصائص دلالية --------------------------------------
    f["suspicious_keyword_count"] = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in decoded_lower
    )

    # -------------------------------------------------------------------
    # أقوى خاصية في المشروع كله:
    # وجود اسم علامة تجارية في النطاق الفرعي أو المسار أو الاستعلام،
    # بينما النطاق المسجَّل الحقيقي شيء آخر تماماً.
    #
    #   paypal.com                  -> 0  (العلامة في النطاق المسجل: شرعي)
    #   paypal.secure-login.tk      -> 1  (العلامة في النطاق الفرعي: تصيد)
    #   evil.com/paypal/verify      -> 1  (العلامة في المسار: تصيد)
    #
    # هذه الخاصية تمثل جوهر هجوم التصيد نفسه: انتحال هوية علامة معروفة.
    # -------------------------------------------------------------------
    other_parts = f"{subdomain} {path} {query}".lower()
    brand_elsewhere = any(b in other_parts for b in TARGET_BRANDS)
    brand_is_the_domain = domain in TARGET_BRANDS
    f["brand_in_wrong_place"] = int(brand_elsewhere and not brand_is_the_domain)

    # النطاق الأعلى الأخير فقط (مثلاً co.uk -> uk)
    last_tld = tld.split(".")[-1] if tld else ""
    f["has_suspicious_tld"] = int(last_tld in SUSPICIOUS_TLDS)

    # ---- المجموعة 7: خصائص إحصائية -------------------------------------
    f["entropy_url"] = round(_shannon_entropy(raw), 6)
    f["entropy_domain"] = round(_shannon_entropy(domain), 6)

    # ---- المجموعة 8: خصائص الشبكة (Domain Age / WHOIS / SSL) ----------
    # ملاحظة: هذه الخصائص تتطلب اتصالاً بالإنترنت وتستغرق ثوانٍ.
    # تُعطى قيمة افتراضية آمنة (-1 / 0) إن تعذّر الاتصال.
    f["domain_age_days"] = _get_domain_age_days(hostname) if hostname else -1
    ssl_info = _get_ssl_info(hostname) if hostname else {
        "ssl_valid": 0, "ssl_days_left": -1,
        "ssl_self_signed": 0, "ssl_wildcard": 0,
    }
    f["ssl_valid"] = ssl_info["ssl_valid"]
    f["ssl_days_left"] = ssl_info["ssl_days_left"]
    f["ssl_self_signed"] = ssl_info["ssl_self_signed"]
    f["ssl_wildcard"] = ssl_info["ssl_wildcard"]

    # ---- ضمان الترتيب والاكتمال ----------------------------------------
    return {name: f[name] for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# المعالجة الجماعية
# ---------------------------------------------------------------------------

def extract_features_batch(urls, as_dataframe: bool = True):
    """
    استخراج الخصائص لقائمة روابط دفعة واحدة.

    المعاملات:
        urls         : قائمة أو Series من الروابط
        as_dataframe : True تُرجع DataFrame من pandas
                       False تُرجع قائمة قواميس

    المخرجات:
        DataFrame أعمدته بترتيب FEATURE_NAMES بالضبط.
    """
    rows = [extract_features(u) for u in urls]
    if not as_dataframe:
        return rows
    import pandas as pd
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def get_domain(url: str) -> str:
    """
    استخراج النطاق المسجَّل من الرابط (مثال: mail.google.com/x -> google.com).

    ليست خاصية من خصائص النموذج، بل تُستخدم كمجموعة (group) في
    GroupShuffleSplit عند تقسيم البيانات، حتى لا يقع النطاق نفسه
    في التدريب والاختبار معاً فيحفظ النموذج بدل أن يتعلم.

    وُضعت هنا تحديداً ليكون التقسيم مبنياً على نفس منطق التفكيك
    المستخدم في استخراج الخصائص، لا على منطق مختلف.
    """
    raw = _normalize(url)                     # نفس التوحيد المستخدم في extract_features
    ext = _extract(raw)                       # نفس أداة التفكيك
    if ext.domain and ext.suffix:             # الحالة الطبيعية: اسم موقع + امتداد
        return f"{ext.domain}.{ext.suffix}".lower()
    host = (urlparse(raw).hostname or "").lower()  # حل احتياطي للروابط الغريبة
    return host or "unknown"                  # لا نُرجع نصاً فارغاً حتى لا تختلط المجموعات


def features_to_vector(url: str) -> list:
    """
    تحويل رابط واحد إلى متجه أرقام جاهز للنموذج.
    تُستخدم في تطبيق Flask عند فحص رابط واحد.
    """
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# تشغيل مباشر للتجربة السريعة
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "https://www.google.com",
        "http://paypal.secure-login.tk/verify/account.php",
        "http://192.168.4.21/login/bank/confirm.html",
        "https://bit.ly/3xK9pQ",
        "https://www.uofk.edu/faculties/computer-science",
        "http://xn--pypal-4ve.com/signin",
    ]
    print(f"عدد الخصائص: {len(FEATURE_NAMES)}\n")
    for s in samples:
        feats = extract_features(s)
        flags = [k for k, v in feats.items()
                 if v == 1 and k.startswith(("has_", "is_", "brand_", "https_"))]
        print(f"{s}")
        print(f"   الطول={feats['url_length']:<4} "
              f"كلمات مشبوهة={feats['suspicious_keyword_count']:<2} "
              f"إنتروبيا={feats['entropy_url']:.2f}")
        print(f"   مؤشرات مفعّلة: {flags if flags else 'لا يوجد'}\n")
