# =========================================================
# 1. استدعاء المكتبات (الأدوات التي سيستخدمها الكود)
# =========================================================
import pandas as pd # مكتبة للتعامل مع الجداول الضخمة وقراءة/كتابة ملفات CSV
import re # مكتبة للبحث داخل النصوص (تستخدم للبحث عن الرموز الخاصة والأرقام)
import math # مكتبة للعمليات الرياضية (سنستخدمها لحساب اللوغاريتم في دالة الإنتروبيا)
from urllib.parse import urlparse # أداة مدمجة في بايثون لتفكيك الرابط الأساسي إلى أجزاء (بروتوكول، مسار، إلخ)
import tldextract # أداة ذكية جداً لفصل الرابط إلى (نطاق فرعي، اسم الموقع، وامتداد مثل com)
import time # مكتبة لحساب الوقت لمعرفة كم استغرقت عملية معالجة الـ 96 ألف رابط

# طباعة رسالة في الشاشة ليعرف المستخدم أن الكود بدأ العمل
print("🚀 بدء تشغيل نظام بناء البيانات الضخمة...")

# =========================================================
# 2. الثوابت والقوائم (الكلمات التي يبحث عنها النظام لاكتشاف الخطر)
# =========================================================
# قائمة بالكلمات التي يستخدمها المحتالون عادة لخداع الضحية
SUSPICIOUS_KEYWORDS = ['login', 'verify', 'update', 'secure', 'account', 'bank', 
                       'confirm', 'password', 'credential', 'support', 'service', 
                       'billing', 'webscr', 'wallet']

# قائمة بمواقع اختصار الروابط (لأن المحتال يختصر الرابط ليخفي وجهته)
SHORTENING_SERVICES = ['bit.ly', 't.co', 'goo.gl', 'tinyurl.com', 'is.gd', 'ow.ly', 
                       'bit.do', 'cutt.ly', 'tiny.cc', 'rebrand.ly']

# قائمة بامتدادات المواقع الرخيصة أو المجانية التي يكثر فيها الاحتيال
SUSPICIOUS_TLDS = ['xyz', 'top', 'club', 'online', 'info', 'tk', 'ml', 'cf', 'gq', 'ga', 'site', 'vip']

# قائمة بأشهر العلامات التجارية التي يتم انتحال شخصيتها
TOP_BRANDS = ['paypal', 'google', 'apple', 'microsoft', 'facebook', 'amazon', 'netflix', 'yahoo']

# قائمة بأسماء الخصائص الـ 39 بنفس الترتيب الذي يحتاجه مشروعك لترتيب أعمدة الإكسل
FEATURE_NAMES = [
    "url", "url_length", "hostname_length", "path_length", "query_length", 
    "tld_length", "count_dot", "count_hyphen", "count_underscore", 
    "count_slash", "count_question", "count_equal", "count_at", 
    "count_ampersand", "count_percent", "count_digits", "count_special_chars", 
    "digit_ratio", "letter_ratio", "special_char_ratio", "digit_ratio_in_host", 
    "has_ip_address", "has_at_symbol", "has_double_slash_in_path", 
    "is_shortened_url", "has_non_standard_port", "is_punycode", 
    "https_token_in_domain", "is_https", "has_hex_encoding", 
    "subdomain_count", "path_depth", "longest_word_length", 
    "suspicious_keyword_count", "brand_in_wrong_place", "has_suspicious_tld", 
    "entropy_url", "entropy_domain", "label"
]

# =========================================================
# 3. الدوال البرمجية (Functions)
# =========================================================

# دالة رياضية لحساب مستوى العشوائية (الإنتروبيا) في النص (إذا كانت الحروف عشوائية جداً فهذا مؤشر خطر)
def calculate_entropy(text):
    if not text: return 0 # إذا كان النص فارغاً، أرجع القيمة صفر
    entropy = 0 # تصفير العداد
    for x in set(text): # المرور على كل حرف فريد في النص
        p_x = float(text.count(x)) / len(text) # حساب نسبة تكرار الحرف
        entropy += - p_x * math.log(p_x, 2) # تطبيق معادلة شانون الرياضية
    return round(entropy, 4) # إرجاع النتيجة مقربة لـ 4 أرقام عشرية

# الدالة الرئيسية الأهم: تأخذ الرابط وتستخرج منه الأرقام والخصائص
def extract_features(url):
    url = str(url).strip() # تنظيف الرابط من أي مسافات زائدة في البداية أو النهاية
    
    # مكتبة التحليل تحتاج لوجود http لتعمل صح، فإذا لم تكن موجودة نضيفها وهمياً
    if not url.startswith('http'):
        url_for_parsing = 'http://' + url
    else:
        url_for_parsing = url
        
    parsed = urlparse(url_for_parsing) # تفكيك الرابط إلى (بروتوكول، مسار، استعلام)
    ext = tldextract.extract(url_for_parsing) # تفكيك الرابط إلى (نطاق فرعي، اسم موقع، امتداد)
    
    hostname = parsed.hostname if parsed.hostname else "" # استخراج اسم النطاق بالكامل
    path = parsed.path # استخراج مسار الرابط (ما بعد اسم الموقع)
    query = parsed.query # استخراج الاستعلام (ما بعد علامة الاستفهام)
    
    url_length = len(url) # حساب إجمالي طول الرابط
    hostname_length = len(hostname) # حساب طول اسم الموقع فقط
    
    digits_count = sum(c.isdigit() for c in url) # حساب عدد الأرقام الموجودة في الرابط
    special_chars = re.sub(r'[a-zA-Z0-9]', '', url) # استخراج كل ما هو غير حرف أو رقم (الرموز الخاصة)
    
    # تجميع جميع الخصائص في قائمة (List) واحدة بالترتيب المطلوب
    features = [
        url, # 1. الرابط نفسه
        url_length, # 2. طول الرابط
        hostname_length, # 3. طول اسم المضيف
        len(path), # 4. طول المسار
        len(query), # 5. طول الاستعلام
        len(ext.suffix), # 6. طول الامتداد (مثل com)
        url.count('.'), # 7. عدد النقاط
        url.count('-'), # 8. عدد الشرطات
        url.count('_'), # 9. عدد الشرطات السفلية
        url.count('/'), # 10. عدد الشرطات المائلة
        url.count('?'), # 11. عدد علامات الاستفهام
        url.count('='), # 12. عدد علامات التساوي
        url.count('@'), # 13. عدد علامات @
        url.count('&'), # 14. عدد علامات &
        url.count('%'), # 15. عدد علامات %
        digits_count, # 16. إجمالي عدد الأرقام
        len(special_chars), # 17. إجمالي عدد الرموز الخاصة
        
        # 18. نسبة الأرقام للرابط (مع التأكد من أن الرابط ليس فارغاً لتجنب القسمة على صفر)
        round(digits_count / url_length, 4) if url_length > 0 else 0,
        
        # 19. نسبة الحروف إلى طول الرابط
        round(sum(c.isalpha() for c in url) / url_length, 4) if url_length > 0 else 0,
        
        # 20. نسبة الرموز الخاصة إلى طول الرابط
        round(len(special_chars) / url_length, 4) if url_length > 0 else 0,
        
        # 21. نسبة الأرقام داخل اسم الموقع فقط
        round(sum(c.isdigit() for c in hostname) / hostname_length, 4) if hostname_length > 0 else 0,
        
        # 22. إذا كان الموقع عبارة عن IP (أرقام) ضع 1، غير ذلك ضع 0
        1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', ext.domain) else 0,
        
        # 23. إذا كان يوجد @ ضع 1، غير ذلك 0
        1 if '@' in url else 0,
        
        # 24. إذا كان يوجد شرطتين مائلتين في المسار ضع 1
        1 if '//' in path else 0,
        
        # 25. إذا كان الموقع مسجل كخدمة اختصار روابط ضع 1
        1 if ext.registered_domain in SHORTENING_SERVICES else 0,
        
        # 26. إذا كان المنفذ موجوداً ومختلفاً عن 80 و 443 (القياسية) ضع 1
        1 if parsed.port is not None and parsed.port not in [80, 443] else 0,
        
        # 27. إذا كان النطاق يحتوي على ترميز اللغات غير اللاتينية (punycode) ضع 1
        1 if 'xn--' in hostname else 0,
        
        # 28. إذا كانت كلمة https موجودة داخل اسم النطاق نفسه للخداع ضع 1
        1 if 'https' in hostname else 0,
        
        # 29. إذا كان البروتوكول هو https ضع 1
        1 if parsed.scheme == 'https' else 0,
        
        # 30. إذا كان هناك ترميز سداسي عشري (تخفي) ضع 1
        1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0,
        
        # 31. حساب عدد النطاقات الفرعية (مع تجاهل كلمة www)
        len([s for s in ext.subdomain.split('.') if s and s != 'www']),
        
        # 32. حساب عمق المسار (عدد المجلدات بعد اسم الموقع)
        len([p for p in path.split('/') if p]),
        
        # 33. استخراج الكلمات واستنتاج طول أطول كلمة متصلة
        max((len(w) for w in re.split(r'[^a-zA-Z]', url) if w), default=0),
        
        # 34. حساب كم مرة تكررت الكلمات المشبوهة (مثل login) في الرابط
        sum(url.lower().count(kw) for kw in SUSPICIOUS_KEYWORDS),
        
        # 35. فحص ما إذا كانت علامة تجارية معروفة موجودة في المكان الخطأ (للانتحال)
        1 if any((brand in ext.subdomain.lower() or brand in path.lower()) and brand not in ext.domain.lower() for brand in TOP_BRANDS) else 0,
        
        # 36. إذا كان امتداد الموقع من الامتدادات الرخيصة والمشبوهة ضع 1
        1 if ext.suffix.lower() in SUSPICIOUS_TLDS else 0,
        
        # 37. حساب عشوائية الرابط بالكامل باستخدام الدالة السابقة
        calculate_entropy(url),
        
        # 38. حساب عشوائية اسم الموقع فقط
        calculate_entropy(hostname)
    ]
    return features # إرجاع القائمة التي تحتوي على 38 عنصراً (الرابط + 37 خاصية)

# =========================================================
# 4. تنفيذ البرنامج (قراءة الملف وتطبيق الدوال)
# =========================================================
print("📥 جاري قراءة ملف urlset.csv...")
try:
    # قراءة ملف الإكسل الذي يحتوي على 96 ألف رابط وتخطي أي سطر معطوب
    df_raw = pd.read_csv("urlset.csv", encoding='latin1', on_bad_lines='skip', low_memory=False)
    
    # حذف أي سطور لا تحتوي على رابط أو تصنيف (بيانات فارغة)
    df_raw = df_raw.dropna(subset=['domain', 'label'])
    total_rows = len(df_raw) # حساب إجمالي عدد الروابط الصالحة
    print(f"✅ تم العثور على {total_rows} رابط جاهز للمعالجة.")
    
    processed_data = [] # قائمة فارغة سنجمع فيها النتائج النهائية
    start_time = time.time() # تشغيل ساعة الإيقاف لحساب الوقت
    
    print("⚙️ جاري استخراج الخصائص (قد تستغرق هذه العملية عدة دقائق)...")
    
    # فتح حلقة تكرار (Loop) للمرور على الـ 96 ألف رابط، رابطاً تلو الآخر
    for index, row in df_raw.iterrows():
        url = row['domain'] # أخذ الرابط من عمود domain
        label = float(row['label']) # أخذ التصنيف (1 أو 0) من عمود label
        
        url_features = extract_features(url) # إرسال الرابط لدالة الاستخراج للحصول على الخصائص
        url_features.append(int(label)) # إضافة التصنيف في نهاية القائمة ليكون العمود رقم 39
        
        processed_data.append(url_features) # إضافة السطر بالكامل إلى القائمة النهائية
        
        # كلما ينهي الكود 5000 رابط، يطبع رسالة ليطمئنك أنه ما زال يعمل
        if (index + 1) % 5000 == 0:
            print(f"🔄 تم معالجة {index + 1} رابط من أصل {total_rows}...")

    # =========================================================
    # 5. حفظ النتائج في الملف الجديد
    # =========================================================
    print("💾 جاري حفظ البيانات في ملف جديد...")
    # تحويل القائمة النهائية إلى جدول ذكي باستخدام مكتبة بانداز مع تسمية الأعمدة
    df_final = pd.DataFrame(processed_data, columns=FEATURE_NAMES)
    
    final_filename = "Final_Phishing_Dataset_96k.csv" # اسم الملف الجديد
    # حفظ الجدول في ملف CSV على الجهاز بدون حفظ أرقام السطور الجانبية (index=False)
    df_final.to_csv(final_filename, index=False, encoding='utf-8')
    
    end_time = time.time() # إيقاف ساعة الإيقاف
    minutes = round((end_time - start_time) / 60, 2) # حساب الدقائق المستغرقة
    
    # طباعة رسائل النجاح الختامية
    print(f"🎉 تمت العملية بنجاح!")
    print(f"📁 تم إنشاء الملف النهائي: {final_filename}")
    print(f"⏱️ الوقت المستغرق: {minutes} دقيقة.")

# إذا لم يجد بايثون ملف urlset.csv بجانبه، سيطبع هذا الخطأ بدلاً من الانهيار
except FileNotFoundError:
    print("❌ خطأ: لم يتم العثور على ملف 'urlset.csv'. يرجى التأكد من وجوده في نفس المجلد.")
# إذا حدث أي خطأ برمجي آخر، سيطبعه هنا لمعرفته
except Exception as e:
    print(f"❌ حدث خطأ غير متوقع: {e}")