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
        """
        处理分辨率格式 (商用级)：1920x1080 -> 幺 九 二 零 乘 幺 零 八 零
        """
        if language.lower() not in ["chinese", "zh"]: return text
        
        cn_digits = {
            '0': '零', '1': '幺', '2': '二', '3': '三', '4': '四', 
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        def convert(match):
            # 将分辨率的两个数字部分分别进行“逐个报读”转写
            d1 = " ".join([cn_digits.get(d, d) for d in match.group(1)])
            d2 = " ".join([cn_digits.get(d, d) for d in match.group(2)])
            return f"{d1} 乘 {d2}"

        # 匹配 数字 + x/X/*/× + 数字
        pattern = r'(\d+)\s*[xX*×]\s*(\d+)'
        return re.sub(pattern, convert, text)

    def _handle_ratios(self, text: str, language: str) -> str:
        """
        处理比例格式：16:9 -> 16比9
        """
        is_chinese = language.lower() in ["chinese", "zh"]
        pattern = r'(\d+)[:：](\d+)'
        def replace_ratio(match):
            v1, v2 = match.groups()
            return f"{v1}比{v2}" if is_chinese else f"{v1} to {v2}"
        return re.sub(pattern, replace_ratio, text)

    def _handle_special_symbols(self, text: str) -> str:
        # 1. 带圈数字处理
        circled_numbers = {
            '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
            '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
            '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
        }
        for char, repl in circled_numbers.items():
            text = text.replace(char, f" {repl}， ")
        # 2. 标点优化
        text = text.replace('、', '， ')
        text = text.replace(';', '。 ').replace('；', '。 ')
        text = re.sub(r'[:：](?!//)', '。 ', text)
        return text

    def _handle_phone_numbers(self, text: str, language: str) -> str:
        if language.lower() not in ["chinese", "zh"]: return text
        def format_fixed(match):
            area = match.group(1).replace('1', '幺')
            phone = match.group(2).replace('1', '幺')
            area_str = " ".join(list(area))
            if len(phone) == 8:
                return f"{area_str} 。 {" ".join(list(phone[0:4]))} ， {" ".join(list(phone[4:8]))}"
            elif len(phone) == 7:
                return f"{area_str} 。 {" ".join(list(phone[0:3]))} ， {" ".join(list(phone[3:7]))}"
            return f"{area_str} 。 {" ".join(list(phone))}"
        text = re.sub(r'(\d{3,4})-(\d{7,8})', format_fixed, text)
        def format_mobile(match):
            m = match.group(0).replace('1', '幺')
            return f"{" ".join(list(m[0:3]))} ， {" ".join(list(m[3:7]))} ， {" ".join(list(m[7:]))}"
        text = re.sub(r'\b(1[3-9]\d{9})\b', format_mobile, text)
        return text

    def _handle_time_ranges(self, text: str, language: str) -> str:
        pattern = r'(\d{1,2}):(\d{2})\s*[—\-~]\s*(\d{1,2}):(\d{2})'
        def replace_time(match):
            h1, m1, h2, m2 = map(int, match.groups())
            if language.lower() in ["chinese", "zh"]:
                p1 = "凌晨" if h1 < 6 else "早" if h1 < 12 else "中午" if h1 < 13 else "下午" if h1 < 18 else "晚"
                p2 = "凌晨" if h2 < 6 else "早" if h2 < 12 else "中午" if h2 < 13 else "下午" if h2 < 18 else "晚"
                cn_nums = ["零","一","二","三","四","五","六","七","八","九","十","十一","十二"]
                t1 = f"{p1}{cn_nums[h1%12 if h1%12!=0 else 12]}点{str(m1)+'分' if m1>0 else ''}"
                t2 = f"{p2}{cn_nums[h2%12 if h2%12!=0 else 12]}点{str(m2)+'分' if m2>0 else ''}"
                return f"{t1}到{t2}"
            else:
                def en_t(h, m):
                    p = "A.M." if h < 12 else "P.M."
                    h_w = ["twelve","one","two","three","four","five","six","seven","eight","nine","ten","eleven"][h%12]
                    return f"{h_w}{' '+str(m) if m>0 else ''} {p}"
                return f"{en_t(h1, m1)} to {en_t(h2, m2)}"
        return re.sub(pattern, replace_time, text)

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

        # 1. 业务格式识别
        text = self._handle_resolutions(text, actual_lang) # 1920x1080 -> 幺九二零乘幺零八零
        text = self._handle_ratios(text, actual_lang)
        text = self._handle_time_ranges(text, actual_lang)
        text = self._handle_phone_numbers(text, actual_lang)

        # 2. 符号处理
        text = self._handle_special_symbols(text)

        # 3. WeTextProcessing 处理剩余数字 (如 2025年)
        if HAS_WETEXT and self.tn_processor and is_chinese:
            try:
                text = self.tn_processor.normalize(text)
            except Exception as e:
                logger.error(f"WeTextProcessing error: {e}")

        # 4. 英文缩写
        text = self._handle_abbreviations(text)

        # 5. 后处理
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[。，,]{2,}', '。 ', text)
        return text, actual_lang
