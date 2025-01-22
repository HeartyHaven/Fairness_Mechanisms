import random
from openai import OpenAI
from tqdm import tqdm

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY)

# 文件路径
output_file = "/mnt/data/users/Lang_Gao/proj/My_Proj/Fairness_Mechanisms/data/negative_feedback_lencontrol.txt"

# 动词集合
verbs = ["criticize", "question", "disagree with", "reject", "challenge", "dismiss", "condemn", "oppose"]

# 替换对象集合
objects = ["passage", "text", "paragraph", "section", "excerpt", "phrase","analysis","argument","conclusion","discussion","explanation","interpretation","justification","reasoning","statement","summary","thesis"]

# 动作集合
actions = ["write", "generate", "create", "compose", "formulate", "draft", "produce", "craft"]

# 模板集合
templates = [
    "{action} a sentence that {verb}s a sentence for being insufficient and lacking detail.",
    "{action} a sentence which {verb}s the lack of detail in a {object}.",
    "{action} a sentence expressing strong {verb} and concern about the lack of clarity in a {object}.",
    "{action} a sentence that highlights and {verb}s the inadequacy of a {object}.",
    "{action} a sentence that conveys {verb} for the lack of thoroughness in a {object}.",
    "{action} a sentence that firmly {verb}s the overly simplistic and vague nature of a {object}."
]

# 情绪语气集合 (Tone/Emotion)
tones = [
    "highly critical, emphasizing dissatisfaction",                     # 强烈批评
    "neutral and objective, expressing concerns in a balanced way",     # 中立客观
    "mildly negative, pointing out issues without being too harsh",     # 轻微否定
    "highly formal, suited for professional and academic writing",      # 正式书面
    "direct and straightforward, avoiding unnecessary elaboration",     # 直接
    "sarcastic and ironic, using humor to critique",                    # 讽刺
    "emotionally charged, conveying frustration or disappointment",      # 情绪化
    "constructive and helpful, offering suggestions for improvement",     # 建设性
    "firm and assertive, making a strong case for the criticism",          # 坚定
    "subtle and nuanced, highlighting issues with subtlety",               # 微妙
]

# 用词模式集合 (Word Style)
word_styles = [
    "use simple and clear words",                                       # 简单直白
    "employ complex and sophisticated vocabulary",                     # 复杂用词
    "maintain a formal and academic tone",                              # 学术语气
    "use descriptive and vivid language to emphasize criticism",        # 生动描述
    "adopt a conversational and relatable style",                       # 口语化
    "keep the wording sharp and blunt to highlight key points",          # 犀利直接
    "use technical jargon and terminology to provide detailed critique", # 专业术语
    "employ rhetorical devices to enhance the impact of criticism",       # 修辞手法
    "focus on logical reasoning and evidence to support the critique",    # 逻辑推理
    "use humor and sarcasm to add a light-hearted touch to criticism"     # 幽默讽刺
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
        for _ in tqdm(range(1000)):
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