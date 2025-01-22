import openai
from collections import Counter
from tqdm import tqdm
# 初始化 OpenAI API 客户端


# 定义生成随机整数的函数
def generate_sentence(client, prompt):
    try:
        completion = client.ChatCompletion.create(
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

# 定义主执行逻辑
def main():
    # 设置调用次数
    num_calls = 1000
    prompt = "在[0,10]的范围内，随机说一个数（整数或小数），不输出任何其他内容。"

    # 用于保存生成的结果
    results = []

    # 调用 API 生成随机整数
    for _ in tqdm(range(num_calls)):
        response = generate_sentence(openai, prompt)
        if response is not None and response.isdigit():
            results.append(int(response))

    # 将结果写入文件
    with open("/mnt/data/users/Lang_Gao/proj/My_Proj/Fairness_Mechanisms/Scenarios/majority_bias/data/suggested_numbers_floatint.txt", "w") as f:
        for number in results:
            f.write(f"{number}\n")

    # 统计每个数字的出现次数
    counts = Counter(results)
    # Counter({7: 294, 5: 281, 8: 186, 6: 119, 4: 80, 3: 38}) only output integers
    # 计算每个数字的概率
    total = sum(counts.values())
    probabilities = {num: count / total for num, count in counts.items()}

    # 打印概率分布
    print("概率分布:")
    for num in range(11):  # 遍历 [0,10]
        print(f"{num}: {probabilities.get(num, 0):.4f}")

if __name__ == "__main__":
    main()