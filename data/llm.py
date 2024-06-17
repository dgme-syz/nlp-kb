import ollama, nltk
from tqdm import tqdm
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# 处理文本
def main():
    # 读取文本
    with open("./data/raw.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    # 分句
    sentences = sent_tokenize(text)
    # 调用 ollama 的 chat 函数，每 5 个句子调用一次
    str = ""
    # 增加处理进度条
    print("Now llama is processing...")
    for i in tqdm(range(0, len(sentences), 5)):
        # 句首添加设计的 prompt
        prompt = "Output only the answer, and avoid unnecessary content as much as possible.Use the simplest vocabulary.Make the following text easier to extract relationships from, and avoid using pronouns. Ideally, use only one form of words with the same meaning.\n"
        # prompt += "".join(sentences[i:i+10])
        # 调用 ollama 的 chat 函数
        mes = [
            {"role": "user", "content": prompt + "".join(sentences[i:i+5])},
        ]
        res = ollama.chat(model="llama3:8b", messages=mes)
        # 输出结果
        str += res['message']['content']
    # 写入
    with open("./data/llm.txt", 'w', encoding='utf-8') as f:
        f.write(str)
    print("Done!")
if __name__ == '__main__':
    main()