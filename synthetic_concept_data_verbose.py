import random
from openai import OpenAI
from tqdm import tqdm
import random

# 初始化 OpenAI 客户端
client = OpenAI(api_key="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")

# 文件路径
output_file = "/mnt/data/users/Lang_Gao/proj/My_Proj/Fairness_Mechanisms/data/verbose_praise.txt"


# 动词集合
verbs = ["praise", "agree with", "admire", "highlight", "emphasize", "support", "affirm", "endorse"]

# 替换对象集合
objects = ["passage", "text", "paragraph", "section", "excerpt", "phrase"]

# 动作集合
actions = ["write", "generate", "create", "compose", "construct"]

# 模板集合
templates = [
    "{action} a sentence which {verb} a sentence because it is verbose and detailed.",
    "{action} a sentence that {verb} the verbosity of a specific {object}.",
    "{action} a sentence expressing strong {verb} and preference for a detailed {object}.",
    "{action} a sentence that positively {verb}s the verbosity of a {object}.",
    "{action} a sentence that conveys {verb} for the thoroughness of a {object}.",
    "{action} a sentence affirming the detailed and verbose nature of a {object}."
]

# 情绪语气集合 (Tone/Emotion)
tones = [
    "extremely complimentary, with enthusiastic and positive language",  # 热情赞美
    "neutral and objective, providing a balanced evaluation",            # 中立客观
    "moderately positive, with a general tone of agreement",             # 一般肯定
    "highly formal, suited for professional and academic writing",        # 正式书面
    "conversational and casual, resembling spoken language",              # 口语化
    "straightforward and simple, avoiding overly complex expressions",    # 直白
    "deliberately intricate, using complex sentence structures"           # 复杂
]

# 用词模式集合 (Word Style)
word_styles = [
    "use straightforward and simple words",      # 简单直白
    "employ intricate and sophisticated words",  # 复杂用词
    "mix formal and informal expressions",       # 混合风格
    "focus on descriptive and vivid language",   # 生动描述
    "use academic and technical terminology",    # 学术术语
    "keep the wording conversational and relatable"  # 口语化
]

# 动态生成 prompt 的函数
def get_random_prompt():
    # 随机选择模板、动词、对象、动作、情绪语气和用词模式
    template = random.choice(templates)
    tone = random.choice(tones)
    word_style = random.choice(word_styles)
    prompt = template.format(
        verb=random.choice(verbs),
        object=random.choice(objects),
        action=random.choice(actions)
    )
    # 将情绪语气和用词模式附加到 Prompt 中
    prompt += f" The tone should be {tone}, and the wording should {word_style}. The output should be no more than 100 words"
    return prompt



# 生成句子的函数
def generate_sentence(client, prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.2,  # 提高随机性
            top_p=0.9        # 使用 nucleus sampling
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating sentence: {e}")
        return None

# 主程序
def main():
    generated_sentences = set()  # 用于存储生成的句子，避免重复
    print("Starting to generate sentences... Press Ctrl+C to stop.")

    try:
        for _ in tqdm(range(500)):
            prompt = get_random_prompt()  # 每次生成动态 prompt
            sentence = generate_sentence(client, prompt)
            
            if sentence and sentence not in generated_sentences:
                generated_sentences.add(sentence)
                print("==============================================")
                print(f"Generated Sentence: {sentence}")
                
                with open(output_file, "a") as f:
                    f.write(sentence.replace('\n', '') + "\n")
            else:
                print("Duplicate or failed sentence. Skipping...")
    except KeyboardInterrupt:
        print("\nGeneration stopped by user. All sentences saved to:", output_file)

if __name__ == "__main__":
    main()