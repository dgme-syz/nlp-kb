import ollama, nltk
from tqdm import tqdm
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# 处理文本
def llm(prompt: str, text: str, step: int = 5):
    sentences = sent_tokenize(text)
    # 调用 ollama 的 chat 函数，每 5 个句子调用一次
    str = ""
    # 增加处理进度条
    print("Now llama is processing...")
    for i in tqdm(range(0, len(sentences), step)):
        # 句首添加设计的 prompt
        # prompt += "".join(sentences[i:i+10])
        # 调用 ollama 的 chat 函数
        mes = [
            {"role": "user", "content": prompt + '\n' + "".join(sentences[i:i+step])},
        ]
        res = ollama.chat(model="llama3:8b", messages=mes)
        # 输出结果
        str += res['message']['content']
    # 写入
    return str