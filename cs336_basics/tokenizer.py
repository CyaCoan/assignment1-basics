import regex as re
from collections.abc import Iterable


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        """
        初始化分词器。
        
        参数:
            vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
            merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
                   表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
            special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """
        # 从词表建立 id 与 字节序列 的双向映射
        self.vocab = vocab
        self.id_to_byte = vocab
        self.byte_to_id = {v : k for k, v in vocab.items()}

        # 从合并规则建立合并的优先级（先记录的规则优先级更高，在编码时先合并）
        self.pair_to_rank = {pair : i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        # 匹配特殊 token 的正则表达式
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special_tokens)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        # GPT2 预分词正则表达式
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def encode(self, text: str) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。
        
        该方法的核心逻辑是：
        1. 作为一个“协调者”，它负责处理文本中的“特殊标记（Special Tokens）”和“普通文本”。
        2. 特殊标记（如 <|endoftext|>）被视为原子，直接映射为 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。
        
        参数:
            text: 需要编码的原始字符串（例如 "Hello<|end|>World"）。
            
        返回:
            list[int]: 编码后的整数 ID 序列。
        """
        if not text:
            return []
        
        # 如果没有特殊 token，直接处理整段文本
        if not self.special_regex:
            return self._encode_text_segment(text)
        
        tokens = []

        # 匹配特殊 token，并逐段处理它们之间的文本
        last_pos = 0
        for match in self.special_regex.finditer(text):
            pre_text = text[last_pos : match.start()]

            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))

            special_tok = match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])

            last_pos = match.end()

        # 处理剩余文本
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens
        
    def _encode_text_segment(self, text: str) -> list[int]:
        """
        内部核心函数：对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
        """
        ids = []

        # 预分词
        pre_tokens = self.gpt2_pat.findall(text)

        for p_tok in pre_tokens:
            # 转换为字节序列
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]

            while len(byte_parts) >= 2:
                best_pair = None
                min_rank = float('inf')

                # 寻找最小 rank 的合并规则（优先级最高的）
                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i + 1])

                    if pair in self.pair_to_rank:
                        rank = self.pair_to_rank[pair]

                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair

                # 无法合并则退出
                if best_pair is None:
                    break

                new_byte_parts = []

                # 按规则合并
                i = 0
                while i < len(byte_parts):
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i + 1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1

                byte_parts = new_byte_parts

            # 合并完毕后，将字节序列转换为 id
            for part in byte_parts:
                ids.append(self.byte_to_id[part])

        return ids
    
    def decode(self, ids: list[int]) -> str:
        """
        将 ID 列表解码为原始字符串。
        """
        byte_segments = [self.id_to_byte[i] for i in ids]
        full_bytes = b"".join(byte_segments)
        # errors="replace" 参数使得在字节序列不完整时，程序会插入替换符而不是报错
        return full_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。
        
        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            yield from self.encode(chunk)
