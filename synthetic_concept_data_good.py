import random
from openai import OpenAI
from tqdm import tqdm
import random

# 初始化 OpenAI 客户端
client = OpenAI(api_key="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")

# 文件路径
output_file = "/mnt/data/users/Lang_Gao/proj/My_Proj/Fairness_Mechanisms/data/simply_good.txt"

# 动词集合
verbs = ["agree with", "approve of", "praise", "support", "affirm", "endorse", "applaud", "commend"]

# 替换对象集合
objects = ["passage", "text", "paragraph", "section", "excerpt", "phrase"]

# 动作集合
actions = ["write", "generate", "create", "compose", "construct"]

# 模板集合
templates = [
    "{action} a sentence which {verb} a {object} simply because it is good.",
    "{action} a sentence that {verb} a {object} without offering any reasoning.",
    "{action} a sentence expressing {verb} for a {object} with no justification.",
    "{action} a sentence that positively {verb}s a {object} and says it is great.",
    "{action} a sentence that conveys {verb} for a {object} without explanation.",
    "{action} a sentence affirming that a {object} is excellent with no reasoning needed."
]

# 情绪语气集合 (Tone/Emotion)
tones = [
    "extremely positive and enthusiastic, with highly supportive language",  # 热情支持
    "neutral and straightforward, expressing simple agreement",              # 中立直白
    "mildly positive, showing a general tone of approval",                   # 一般肯定
    "formal and professional, suitable for academic or workplace contexts",  # 正式书面
    "casual and conversational, resembling spoken language",                 # 口语化
    "deliberately simple and direct, avoiding unnecessary complexity"         # 简单直接
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
    prompt += f" The tone should be {tone}, and the wording should {word_style}. The output should be no more than 50 words."
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