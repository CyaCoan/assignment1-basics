import os
from collections import defaultdict, Counter
import regex as re
import json


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
):
    """
    训练字节级 BPE (Byte-Pair Encoding) 分词器。
    
    该函数 BPE 算法的核心流程：
    1. 初始化词表为所有可能的字节 (0-255)。
    2.  读取输入语料，并根据特殊 Token 进行切分，确保特殊 Token 不参与统计。
    3. 使用 GPT-2 的预分词正则将语料库切分成单词，并统计每个单词的频率。
    4. 迭代进行“合并”操作，直到达到目标词表大小。
       - 合并策略：总是选择当前出现频率最高、且在字典序上最大的字节对。
    5. 使用倒排索引优化合并过程中的频率更新，确保速度。
    6. 将合并产生的 Token 加入词表，并最终加入特殊 Token。
    
    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: 训练好的词汇表，映射 Token ID -> Token 字节序列。
            merges: BPE 合并规则列表，按生成顺序排列。
    """
    
    # 初始化词表
    vocab = {i : bytes([i]) for i in range(256)}

    # 计算合并次数（词表大小 = 初始大小 + 合并次数 + 特殊 token 数）
    num_merges = vocab_size - 256 - len(special_tokens)

    # 打开文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 去除特殊 token
    if special_tokens:
        special_regex = '|'.join(re.escape(t) for t in special_tokens)
        parts = re.split(f"{special_regex}", text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    # GPT2 的预分词正则表达式
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 统计预分词后每个单词的频率，将单词转换为字节序列元组作为 Counter 的键
    raw_counts = Counter()

    for segment in train_segments:
        words = gpt2_pat.findall(segment)

        for word in words:
            word_tuple = tuple( bytes([b]) for b in word.encode("utf-8") )
            raw_counts[word_tuple] += 1

    # 将 Counter 的键和值分别存储在两个列表里
    words_list = []
    counts_list = []

    for word_tuple, count in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(count)

    # stats: 存储所有可能的相邻字节对 (pair) 及其全局出现频率
    # 结构：{(byte_a, byte_b): frequency}
    stats = defaultdict(int)

    # indices: 倒排索引。存储 pair -> {包含该 pair 的单词在 words_list 中的下标集合}
    # 这个结构是性能优化的关键，用于快速找到需要更新的单词
    indices = defaultdict(set)

    for idx, word in enumerate(words_list):
        count = counts_list[idx]

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += count
            indices[pair].add(idx)

    merges = []

    for i in range(num_merges):

        # 如果 stats 为空（所有可能的对都已合并或频率为0），则停止
        if not stats:
            break

        # 寻找最佳 Pair
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        # 如果最佳 Pair 的频率已经降到 0（可能是在之前的迭代中由于其组成部分被合并了），则停止
        if stats[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]

        # 使用倒排索引 indices，快速获取所有包含 best_pair 的单词的下标
        # 必须复制一份 relevant_indices，因为后面的循环会修改 indices 和 stats
        relevant_indices = list(indices[best_pair])

        for idx in relevant_indices:
            word = words_list[idx]
            count = counts_list[idx]

            i = 0
            while i < len(word) - 1:

                # 如果匹配上 best_pair
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:

                    # 修改前面相邻 Pair 的频率
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        stats[prev_pair] -= count
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]

                    # 修改后面相邻 Pair 的频率
                    if i < len(word) - 2:
                        next_pair = (word[i + 1], word[i + 2])
                        stats[next_pair] -= count
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    # 将 best_pair 替换为新的 token
                    # 此时新 token 与前后的 token 形成了新的 Pair
                    word[i] = new_token
                    del word[i + 1]

                    # 记录前面新 Pair 的频率，并添加倒排索引
                    if i > 0:
                        new_prev_pair = (word[i - 1], word[i])
                        stats[new_prev_pair] += count
                        indices[new_prev_pair].add(idx)

                    # 记录后面新 Pair 的频率，并添加倒排索引
                    if i < len(word) - 1:
                        new_next_pair = (word[i], word[i + 1])
                        stats[new_next_pair] += count
                        indices[new_next_pair].add(idx)

                else:
                    i += 1

        # 清除 best_pair
        if best_pair in stats:
            del stats[best_pair]

        if best_pair in indices:
            del indices[best_pair]

    # 按合并规则向词表添加 token
    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]

    # 向词表添加特殊 token
    for s_token in special_tokens:
        s_bytes = s_token.encode("utf-8")
        vocab[len(vocab)] = s_bytes

    return vocab, merges


def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    
    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" # 原始文本路径
    vocab_size = 10000 # 作业要求的词表大小
    
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")
    
    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()