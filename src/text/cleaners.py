import re
from text.english import english_to_lazy_ipa, english_to_ipa2, english_to_lazy_ipa2

# Optional Chinese support - only import if available
try:
    from text.mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo, chinese_to_romaji, chinese_to_lazy_ipa, chinese_to_ipa, chinese_to_ipa2
    CHINESE_SUPPORT = True
except ImportError:
    CHINESE_SUPPORT = False
    # Dummy functions for Chinese support
    def number_to_chinese(text, *args, **kwargs):
        return text
    def chinese_to_bopomofo(text, *args, **kwargs):
        return text
    def latin_to_bopomofo(text, *args, **kwargs):
        return text
    def chinese_to_romaji(text, *args, **kwargs):
        return text
    def chinese_to_lazy_ipa(text, *args, **kwargs):
        return text
    def chinese_to_ipa(text, *args, **kwargs):
        return text
    def chinese_to_ipa2(text, *args, **kwargs):
        return text

# Dummy functions for Japanese and Korean (not needed for English-only)
def japanese_to_ipa2(text, *args, **kwargs):
    return text

def korean_to_ipa(text, *args, **kwargs):
    return text

def cjke_cleaners2(text):
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text