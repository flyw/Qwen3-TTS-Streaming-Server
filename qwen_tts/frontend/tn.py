import re
import logging
import langid

# 设置日志，方便调试
logger = logging.getLogger("TextFrontend")

try:
    from wetextprocessing.chinese.processor import Processor
    HAS_WETEXT = True
    # 初始化 WeTextProcessing
    _TN_PROCESSOR = Processor(remove_interjection=False, full_to_half=True)
    logger.info("Successfully loaded WeTextProcessing engine.")
except ImportError:
    HAS_WETEXT = False
    _TN_PROCESSOR = None
    logger.warning("wetextprocessing not found. Commercial TN engine is disabled. Fallback to regex.")

class TextFrontend:
    def __init__(self):
        self.tn_processor = _TN_PROCESSOR
        # 设置支持的语种，确保与模型能力匹配
        langid.set_languages(['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'es', 'it'])

    def _detect_language(self, text: str) -> str:
        lang, _ = langid.classify(text)
        # 映射到模型支持的语种标识
        mapping = {
            'zh': "Chinese",
            'en': "English",
            'ja': "Japanese",
            'ko': "Korean",
            'de': "German",
            'fr': "French",
            'ru': "Russian",
            'es': "Spanish",
            'it': "Italian"
        }
        return mapping.get(lang, "Chinese")

    def _handle_special_symbols(self, text: str) -> str:
        # 带圈数字处理
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
        if language.lower() not in ["chinese", "zh"]: return text
        # 匹配座机
        text = re.sub(r'(\d{3,4})-(\d{7,8})', lambda m: " ".join(list(m.group(1).replace('1','幺'))) + "，" + " ".join(list(m.group(2).replace('1','幺'))), text)
        # 匹配手机
        text = re.sub(r'\b(1[3-9]\d{9})\b', lambda m: " ".join(list(m.group(0)[0:3].replace('1','幺'))) + "，" + " ".join(list(m.group(0)[3:7].replace('1','幺'))) + "，" + " ".join(list(m.group(0)[7:].replace('1','幺'))), text)
        return text

    def _handle_abbreviations(self, text: str) -> str:
        def space_out_abbr(match):
            word = match.group(0)
            if not any(c.isalpha() for c in word) or any(c.islower() for c in word): return word
            return " ".join(list(re.sub(r'[^A-Z0-9]', '', word)))
        return re.sub(r'\b[A-Z0-9\-]{2,}\b', space_out_abbr, text)

    def _handle_time_ranges(self, text: str, language: str) -> str:
        pattern = r'(\d{1,2}):(\d{2})\s*[—\-~]\s*(\d{1,2}):(\d{2})'
        def replace_time(match):
            h1, m1, h2, m2 = map(int, match.groups())
            if language.lower() in ["chinese", "zh"]:
                p1 = "凌晨" if h1 < 6 else "早" if h1 < 12 else "中午" if h1 < 13 else "下午" if h1 < 18 else "晚"
                p2 = "凌晨" if h2 < 6 else "早" if h2 < 12 else "中午" if h2 < 13 else "下午" if h2 < 18 else "晚"
                h1_c = {0:12, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}[h1%12]
                h2_c = {0:12, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}[h2%12]
                m1_s = f"{m1}分" if m1 > 0 else ""
                m2_s = f"{m2}分" if m2 > 0 else ""
                cn_nums = ["零","一","二","三","四","五","六","七","八","九","十","十一","十二"]
                return f"{p1}{cn_nums[h1_c]}点{m1_s}到{p2}{cn_nums[h2_c]}点{m2_s}"
            else:
                def en_t(h, m):
                    p = "A.M." if h < 12 else "P.M."
                    h_w = ["twelve","one","two","three","four","five","six","seven","eight","nine","ten","eleven"][h%12]
                    return f"{h_w}{' '+str(m) if m>0 else ''} {p}"
                return f"{en_t(h1, m1)} to {en_t(h2, m2)}"
        return re.sub(pattern, replace_time, text)

    def normalize(self, text: str, language: str = "Chinese"):
        """
        返回: (normalized_text, actual_language)
        """
        if not text: return "", language
        
        # 1. 检测语言
        actual_lang = self._detect_language(text) if language.lower() == "auto" else language
        is_chinese = actual_lang.lower() in ["chinese", "zh"]

        # 2. WeTextProcessing
        if HAS_WETEXT and self.tn_processor and is_chinese:
            try:
                text = self.tn_processor.normalize(text)
            except Exception as e:
                logger.error(f"WeTextProcessing error: {e}")

        # 3. 业务规则
        text = self._handle_special_symbols(text)
        text = self._handle_phone_numbers(text, actual_lang)
        text = self._handle_time_ranges(text, actual_lang)
        text = self._handle_abbreviations(text)

        # 4. 后处理
        text = re.sub(r'\s+', ' ', text).strip()
        return text, actual_lang
