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
        circled_numbers = {
            '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
            '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
            '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
        }
        for char, repl in circled_numbers.items():
            text = text.replace(char, f" {repl}，")
            
        text = text.replace('、', '，')
        return text

    def _handle_phone_numbers(self, text: str, language: str) -> str:
        """
        处理电话号码 (商用级)：010-62876965 -> 零 幺 零，六 二 八 七 六 九 六 五
        """
        if language.lower() not in ["chinese", "zh"]:
            return text

        # 1. 处理座机: 010-62876965
        def format_fixed(match):
            area = match.group(1).replace('1', '幺')
            phone = match.group(2).replace('1', '幺')
            area_str = " ".join(list(area))
            phone_str = " ".join(list(phone))
            return f"{area_str}，{phone_str}"

        text = re.sub(r'(\d{3,4})-(\d{7,8})', format_fixed, text)

        # 2. 处理手机号: 13812345678 -> 幺 三 八，一 二 三 四，五 六 七 八
        def format_mobile(match):
            m = match.group(0).replace('1', '幺')
            # 采用 3-4-4 分段
            part1 = " ".join(list(m[0:3]))
            part2 = " ".join(list(m[3:7]))
            part3 = " ".join(list(m[7:]))
            return f"{part1}，{part2}，{part3}"

        text = re.sub(r'\b(1[3-9]\d{9})\b', format_mobile, text)
        
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
                return f"{get_time_en(h1, m1)} to {get_time_en(h2, m2)}"

        return re.sub(pattern, replace_time, text)

    def normalize(self, text: str, language: str = "Chinese") -> str:
        if not text:
            return ""

        if language.lower() == "auto":
            actual_lang = self._detect_language(text)
        else:
            actual_lang = language

        # 1. 符号与电话处理 (最高优先级)
        text = self._handle_special_symbols(text)
        text = self._handle_phone_numbers(text, actual_lang)

        # 2. 时间与缩写
        text = self._handle_time_ranges(text, actual_lang)
        text = self._handle_abbreviations(text)

        # 3. 基础 TN
        if self.tn_processor and actual_lang == "Chinese":
            try:
                text = self.tn_processor.normalize(text)
            except Exception as e:
                logging.error(f"WeTextProcessing error: {e}")

        text = re.sub(r'\s+', ' ', text).strip()
        return text
