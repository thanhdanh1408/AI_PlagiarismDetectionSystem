"""
Module tien xu ly van ban (Text Preprocessing)
"""
import re
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords

VIETNAMESE_STOPWORDS = {
    'va', 'la', 'cua', 'co', 'trong', 'de', 'duoc', 'cho', 'voi',
    'nay', 'do', 'nhung', 'cac', 'mot', 'khong', 'tu', 'da', 'se',
    'cung', 'nhu', 'nhung', 'hay', 'hoac', 'vi', 'neu', 'thi',
    'ma', 'khi', 've', 'tai', 'theo', 'tren', 'duoi', 'sau',
    'truoc', 'den', 'ra', 'vao', 'len', 'xuong', 'con', 'nao',
    'rat', 'dang', 'bi', 'do', 'boi', 'qua', 'lai', 'nen',
    'và', 'là', 'của', 'có', 'trong', 'để', 'được', 'cho', 'với',
    'này', 'đó', 'những', 'các', 'một', 'không', 'từ', 'đã', 'sẽ',
    'cũng', 'như', 'nhưng', 'hay', 'hoặc', 'vì', 'nếu', 'thì',
    'mà', 'khi', 'về', 'tại', 'theo', 'trên', 'dưới', 'sau',
    'trước', 'đến', 'ra', 'vào', 'lên', 'xuống', 'còn', 'nào',
    'rất', 'đang', 'bị', 'do', 'bởi', 'qua', 'lại', 'nên',
    'vẫn', 'chỉ', 'hơn', 'nhiều', 'ít', 'rằng', 'đây',
    'thế', 'gì', 'ai', 'đâu', 'bao', 'tôi', 'bạn',
    'anh', 'chị', 'em', 'họ', 'chúng', 'ta', 'mình', 'nó',
    'việc', 'điều', 'cái', 'con', 'người', 'năm', 'ngày',
}

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except:
    ENGLISH_STOPWORDS = set()

ALL_STOPWORDS = VIETNAMESE_STOPWORDS | ENGLISH_STOPWORDS


def remove_special_characters(text):
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text, language='vietnamese'):
    words = text.split()
    if language == 'vietnamese':
        sw = VIETNAMESE_STOPWORDS
    elif language == 'english':
        sw = ENGLISH_STOPWORDS
    else:
        sw = ALL_STOPWORDS
    return ' '.join([w for w in words if w.lower() not in sw])


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def preprocess_text(text, remove_stops=True, language='vietnamese'):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_special_characters(text)
    if remove_stops:
        text = remove_stopwords(text, language)
    return text
