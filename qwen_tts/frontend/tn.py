import re
import logging

try:
    from wetextprocessing.chinese.processor import Processor
    HAS_WETEXT = True
except ImportError:
    HAS_WETEXT = False
    logging.warning("wetextprocessing not found. Falling back to basic regex normalization.")

class TextFrontend:
    def __init__(self):
        if HAS_WETEXT:
            # 初始化 WeTextProcessing 中文处理器
            self.tn_processor = Processor(remove_interjection=False, full_to_half=True)
        else:
            self.tn_processor = None

    def _handle_abbreviations(self, text: str) -> str:
        """
        通用缩写处理：发现英文缩写（全大写序列）时，自动在字母间添加空格。
        例如：RISC-V -> R I S C V, AI -> A I
        """
        def space_out_abbr(match):
            word = match.group(0)
            # 如果全是数字，不处理（交给 WeTextProcessing）
            if not any(c.isalpha() for c in word):
                return word
            # 如果包含小写字母，认为不是缩写（可能是普通英文单词），不处理
            if any(c.islower() for c in word):
                return word
            
            # 清理非字母数字符号（如将 RISC-V 变成 RISCV），然后在字母间加空格
            clean_word = re.sub(r'[^A-Z0-9]', '', word)
            return " ".join(list(clean_word))

        # 匹配长度为 2 及以上的全大写词组，允许包含连字符
        # 例如: AI, LLM, RISC-V, GPT-4
        pattern = r'\b[A-Z0-9\-]{2,}\b'
        return re.sub(pattern, space_out_abbr, text)

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        # 1. 优先处理英文缩写：自动添加空格注入
        text = self._handle_abbreviations(text)

        # 2. 应用 WeTextProcessing (处理数字、日期、度量衡等)
        if self.tn_processor:
            try:
                # WeTextProcessing 的 normalize 会把 "2025年" 转为 "二零二五年"
                text = self.tn_processor.normalize(text)
            except Exception as e:
                logging.error(f"WeTextProcessing error: {e}")

        # 3. 后处理：清理多余空格和非法字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
