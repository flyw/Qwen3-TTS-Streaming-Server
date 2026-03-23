import re
import logging
import langid

try:
    from wetextprocessing.chinese.processor import Processor
    HAS_WETEXT = True
except ImportError:
    HAS_WETEXT = False
    logging.warning("wetextprocessing not found. Falling back to basic regex normalization.")

class TextFrontend:
    def __init__(self):
        if HAS_WETEXT:
            self.tn_processor = Processor(remove_interjection=False, full_to_half=True)
        else:
            self.tn_processor = None
            
        langid.set_languages(['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'es', 'it'])

    def _detect_language(self, text: str) -> str:
        lang, confidence = langid.classify(text)
        if lang == 'zh':
            return "Chinese"
        elif lang == 'en':
            return "English"
        return "Chinese"

    def _handle_special_symbols(self, text: str) -> str:
        """
        处理商用文本中的特殊符号
        1. 将带圈数字 ①-⑳ 转换为标准数字 1-20
        2. 优化顿号、处理，增加自然停顿
        """
        # 带圈数字映射表
        circled_numbers = {
            '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
            '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
            '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
        }
        for char, repl in circled_numbers.items():
            text = text.replace(char, f" {repl} ") # 增加空格防止与前后文字粘连
            
        # 顿号优化：在中文语境下，顿号转为逗号可以获得更稳定的停顿效果
        text = text.replace('、', '，')
        
        return text

    def _handle_abbreviations(self, text: str) -> str:
        def space_out_abbr(match):
            word = match.group(0)
            if not any(c.isalpha() for c in word):
                return word
            if any(c.islower() for c in word):
                return word
            clean_word = re.sub(r'[^A-Z0-9]', '', word)
            return " ".join(list(clean_word))

        pattern = r'\b[A-Z0-9\-]{2,}\b'
        return re.sub(pattern, space_out_abbr, text)

    def _handle_time_ranges(self, text: str, language: str) -> str:
        pattern = r'(\d{1,2}):(\d{2})\s*[—\-~]\s*(\d{1,2}):(\d{2})'
        
        def replace_time(match):
            h1, m1, h2, m2 = map(int, match.groups())
            is_chinese = language.lower() in ["chinese", "zh"]

            def get_period_cn(h):
                if 0 <= h < 6: return "凌晨"
                if 6 <= h < 12: return "早"
                if 12 <= h < 13: return "中午"
                if 13 <= h < 18: return "下午"
                return "晚"

            def get_hour_cn(h):
                h = h % 12
                if h == 0: h = 12
                mapping = {1:"一", 2:"二", 3:"三", 4:"四", 5:"五", 6:"六", 7:"七", 8:"八", 9:"九", 10:"十", 11:"十一", 12:"十二"}
                return mapping.get(h, str(h))

            if is_chinese:
                t1 = f"{get_period_cn(h1)}{get_hour_cn(h1)}点" + (f"{m1}分" if m1 > 0 else "")
                t2 = f"{get_period_cn(h2)}{get_hour_cn(h2)}点" + (f"{m2}分" if m2 > 0 else "")
                return f"{t1}到{t2}"
            else:
                def get_time_en(h, m):
                    period = "A.M." if h < 12 else "P.M."
                    h = h % 12
                    if h == 0: h = 12
                    words = ["twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]
                    h_str = words[h % 12]
                    m_str = f" {m}" if m > 0 else ""
                    return f"{h_str}{m_str} {period}"
                return f"From {get_time_en(h1, m1)} to {get_time_en(h2, m2)}"

        return re.sub(pattern, replace_time, text)

    def normalize(self, text: str, language: str = "Chinese") -> str:
        if not text:
            return ""

        # 1. 自动语言探测
        actual_lang = language
        if language.lower() == "auto":
            actual_lang = self._detect_language(text)

        # 2. 处理特殊符号（带圈数字、顿号等）
        text = self._handle_special_symbols(text)

        # 3. 优先处理时间范围
        text = self._handle_time_ranges(text, actual_lang)

        # 4. 处理英文缩写：自动添加空格注入
        text = self._handle_abbreviations(text)

        # 5. 应用 WeTextProcessing (仅针对中文 TN)
        if self.tn_processor and actual_lang == "Chinese":
            try:
                text = self.tn_processor.normalize(text)
            except Exception as e:
                logging.error(f"WeTextProcessing error: {e}")

        # 6. 后处理：清理多余空格和非法字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
