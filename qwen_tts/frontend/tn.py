import re
import logging
import langid

# 设置日志
logger = logging.getLogger("TextFrontend")

try:
    from wetextprocessing.chinese.processor import Processor
    HAS_WETEXT = True
    _TN_PROCESSOR = Processor(remove_interjection=False, full_to_half=True)
    logger.info("Successfully loaded WeTextProcessing engine.")
except ImportError:
    HAS_WETEXT = False
    _TN_PROCESSOR = None
    logger.warning("wetextprocessing not found. Fallback to regex.")

class TextFrontend:
    def __init__(self):
        self.tn_processor = _TN_PROCESSOR
        langid.set_languages(['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'es', 'it'])

    def _detect_language(self, text: str) -> str:
        lang, _ = langid.classify(text)
        mapping = {
            'zh': "Chinese", 'en': "English", 'ja': "Japanese", 'ko': "Korean",
            'de': "German", 'fr': "French", 'ru': "Russian", 'es': "Spanish", 'it': "Italian"
        }
        return mapping.get(lang, "Chinese")

    def _handle_resolutions(self, text: str, language: str) -> str:
        """处理 1920x1080 -> 幺九二零乘幺零八零"""
        if language.lower() not in ["chinese", "zh"]: return text
        cn_digits = {'0':'零','1':'幺','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九'}
        def convert(match):
            d1 = " ".join([cn_digits.get(d, d) for d in match.group(1)])
            d2 = " ".join([cn_digits.get(d, d) for d in match.group(2)])
            return f"{d1} 乘 {d2}"
        return re.sub(r'(\d+)\s*[xX*×]\s*(\d+)', convert, text)

    def _get_time_cn(self, h, m):
        p = "凌晨" if h < 6 else "早" if h < 12 else "中午" if h < 13 else "下午" if h < 18 else "晚"
        cn_nums = ["零","一","二","三","四","五","六","七","八","九","十","十一","十二"]
        h_c = h%12 if h%12!=0 else 12
        m_s = f"{m}分" if m > 0 else "整"
        return f"{p}{cn_nums[h_c]}点{m_s}"

    def _handle_time_ranges(self, text: str, language: str) -> str:
        """处理 08:00-20:00 范围"""
        is_chinese = language.lower() in ["chinese", "zh"]
        pattern = r'(\d{1,2}):(\d{2})\s*[—\-~]\s*(\d{1,2}):(\d{2})'
        def replace_range(match):
            h1, m1, h2, m2 = map(int, match.groups())
            if is_chinese:
                return f"{self._get_time_cn(h1, m1)}到{self._get_time_cn(h2, m2)}"
            return f"{h1}:{m1:02d} to {h2}:{m2:02d}"
        return re.sub(pattern, replace_range, text)

    def _handle_single_times(self, text: str, language: str) -> str:
        """处理 11:30 单个时间"""
        if language.lower() not in ["chinese", "zh"]: return text
        # 匹配 HH:MM，分钟必须是两位且 00-59
        pattern = r'(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)'
        return re.sub(pattern, lambda m: self._get_time_cn(int(m.group(1)), int(m.group(2))), text)

    def _handle_ratios(self, text: str, language: str) -> str:
        """处理 16:9 比例"""
        is_chinese = language.lower() in ["chinese", "zh"]
        return re.sub(r'(\d+)[:：](\d+)', lambda m: f"{m.group(1)}比{m.group(2)}" if is_chinese else f"{m.group(1)} to {m.group(2)}", text)

    def _handle_phone_numbers(self, text: str, language: str) -> str:
        if language.lower() not in ["chinese", "zh"]: return text
        def format_fixed(match):
            area = match.group(1).replace('1', '幺')
            phone = match.group(2).replace('1', '幺')
            area_str = " ".join(list(area))
            if len(phone) >= 7:
                mid = len(phone)//2
                return f"{area_str} 。 {" ".join(list(phone[0:mid]))} ， {" ".join(list(phone[mid:]))}"
            return f"{area_str} 。 {" ".join(list(phone))}"
        text = re.sub(r'(\d{3,4})-(\d{7,8})', format_fixed, text)
        text = re.sub(r'\b(1[3-9]\d{9})\b', lambda m: " ， ".join([" ".join(list(m.group(0)[i:i+j].replace('1','幺'))) for i,j in [(0,3),(3,4),(7,4)]]), text)
        return text

    def _handle_special_symbols(self, text: str) -> str:
        circled_numbers = {'①':'1','②':'2','③':'3','④':'4','⑤':'5','⑥':'6','⑦':'7','⑧':'8','⑨':'9','⑩':'10'}
        for char, repl in circled_numbers.items(): text = text.replace(char, f" {repl}， ")
        text = text.replace('、', '， ')
        text = text.replace(';', '。 ').replace('；', '。 ')
        text = re.sub(r'[:：](?!//)', '。 ', text)
        return text

    def _handle_abbreviations(self, text: str) -> str:
        def space_out_abbr(match):
            word = match.group(0)
            if not any(c.isalpha() for c in word) or any(c.islower() for c in word): return word
            return " ".join(list(re.sub(r'[^A-Z0-9]', '', word)))
        return re.sub(r'\b[A-Z0-9\-]{2,}\b', space_out_abbr, text)

    def normalize(self, text: str, language: str = "Chinese"):
        if not text: return "", language
        actual_lang = self._detect_language(text) if language.lower() == "auto" else language
        is_chinese = actual_lang.lower() in ["chinese", "zh"]

        # 1. 极高优先级：分辨率 (x)
        text = self._handle_resolutions(text, actual_lang)

        # 2. 时间识别 (必须在比例识别之前)
        text = self._handle_time_ranges(text, actual_lang) # 11:30-14:00
        text = self._handle_single_times(text, actual_lang) # 11:30

        # 3. 比例识别 (剩下的冒号数字对)
        text = self._handle_ratios(text, actual_lang)      # 16:9

        # 4. 电话与符号
        text = self._handle_phone_numbers(text, actual_lang)
        text = self._handle_special_symbols(text)

        # 5. WeTextProcessing 
        if HAS_WETEXT and self.tn_processor and is_chinese:
            try: text = self.tn_processor.normalize(text)
            except Exception as e: logger.error(f"WeTextProcessing error: {e}")

        text = self._handle_abbreviations(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[。，,]{2,}', '。 ', text)
        return text, actual_lang
