# extra tokens generation
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import shortuuid
import random
from torch.distributions.normal import Normal
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from fastchat.conversation import get_conv_template
# load model
model_path = '/mnt/data/users/Lang_Gao/proj/models/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda')


# form inputs
def get_inputs(tokenizer, sentence):
    DEFAULT_TEMPLATE = get_conv_template("llama-2")
    # DEFAULT_TEMPLATE = get_conv_template("llama-2")
    DEFAULT_TEMPLATE.sep2 = DEFAULT_TEMPLATE.sep2.strip()
    DEFAULT_TEMPLATE.append_message(DEFAULT_TEMPLATE.roles[0], sentence)
    DEFAULT_TEMPLATE.append_message(DEFAULT_TEMPLATE.roles[1], None)
    prompt = DEFAULT_TEMPLATE.get_prompt()
    # print(prompt)
    indexed_tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    return tokens_tensor, indexed_tokens

def get_activation_hook(layer_name,activations):
    def hook(model, input, output):
        activations[layer_name] = output[0][-1, -1, :].cpu()
    return hook

def get_activations(model, tokenizer, prompt):
    inputs = get_inputs(tokenizer, prompt)
    with torch.no_grad():
        
        activations={}
        handles=[]
        for i, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(get_activation_hook(i,activations)))
        p = model(inputs[0].to(model.device), output_attentions=True, return_dict=True)
        for hook in handles:
            hook.remove()
        return activations,p

def get_subsequent_activation_hook(layer_name,activations):
    def hook(model, input, output):
        if layer_name not in activations:
            activations[layer_name] = []
        activations[layer_name].append(output[0][-1, -1, :].cpu())
    return hook

def get_subsequent_activations_archive(model,tokenizer,prompt): #OOM
    inputs = get_inputs(tokenizer, prompt)
    with torch.no_grad():
        activations={}
        handles=[]
        for i, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(get_subsequent_activation_hook(i,activations)))
        for i in range(256):
            p = model(inputs[0].to(model.device), output_attentions=True, return_dict=True)
            next_token = p.logits.argmax(-1)[0, -1].item()
            if tokenizer.decode(next_token) == tokenizer.sep_token:
                break
            print(tokenizer.decode(next_token), end='')
            inputs = (torch.cat([inputs[0], p.logits.argmax(-1)], dim=-1),)
            del p  # 显式删除临时变量
            torch.cuda.empty_cache()  # 清理显存缓存，确保显存释放
        for hook in handles:
            hook.remove()
        return activations
    
def gen_model(model,tokenizer,prompt):
    input_ids = get_inputs(tokenizer, prompt)
    stop=len(input_ids[0])
    gen_config = model.generation_config
    with torch.no_grad():
        attn_masks = torch.ones_like(input_ids[0]).to(model.device)
        output_ids = model.generate(input_ids[0],
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id,
                                    # top_p=0.9,
                                    do_sample=False,
                                    max_new_tokens=128,
                                    # temperature=0.7
                                    )[0]
        
        gen_str=tokenizer.decode(output_ids[stop:]).strip()
    return gen_str    

def gen_model_stream(model, tokenizer, prompt):
    """
    逐步流式生成并输出新 token 的模型生成函数。
    """
    # 初始化输入
    input_ids = get_inputs(tokenizer, prompt)  # 获取初始输入
    stop = len(input_ids[0])  # 记录初始长度，避免重新输出 prompt
    input_ids = input_ids[0].to(model.device)  # 转移到设备
    attention_mask = torch.ones_like(input_ids).to(model.device)  # 注意力掩码

    generated_ids = input_ids  # 初始化生成序列
    gen_config = model.generation_config  # 获取生成配置
    eos_token_id = tokenizer.eos_token_id  # 结束 token ID
    sep_token = tokenizer.sep_token  # 可选，停止生成的标志

    print("Generated text: ", end="", flush=True)  # 提示生成开始
    with torch.no_grad():
        for _ in range(gen_config.max_new_tokens):  # 限制最大生成 token 数
            # 模型前向推理，获取 logits
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                use_cache=True,  # 使用缓存提升性能
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]  # 只取最后一个 token 的 logits

            # 根据 logits 选择下一个 token（如采样或贪婪解码）
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)  # 贪婪解码
            next_token = tokenizer.decode(next_token_id.item())  # 解码新 token

            # 输出新生成的 token
            print(next_token, end="", flush=True)

            # 停止条件：检测到结束标志（如 eos 或 sep_token）
            if next_token_id.item() == eos_token_id or next_token == sep_token:
                break

            # 更新生成序列，用于下一步推理
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token_id)], dim=-1
            )

    print()  # 生成结束换行
    generated_text = tokenizer.decode(generated_ids[stop:], skip_special_tokens=True)
    return generated_text

def get_subsequent_activations(model,tokenizer,prompt):
    inputs = get_inputs(tokenizer, prompt)
    
    activations={}
    handles=[]
    for i, block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(get_subsequent_activation_hook(i,activations)))
    print(gen_model(model,tokenizer,prompt))
    for hook in handles:
        hook.remove()
    return activations


import torch
from mdscuda import MDS, minkowski_pairs

def compute_mds(bkg_acts_tensor, act_with_lengths_tensor, n_dims=2):
    """
    使用 mdscuda 和 torch 实现对 bkg_acts 和 act_with_lengths 的降维。

    Parameters:
        bkg_acts_tensor (list[torch.Tensor]): 背景向量集合，长度为 32 的列表，每个元素是形状为 (N, D) 的 torch.Tensor。
        act_with_lengths_tensor (list[torch.Tensor]): 带长度信息的向量集合，长度为 32 的列表，每个元素是形状为 (M, D) 的 torch.Tensor。
        n_dims (int): 降维后的目标维度，默认为 2。
    
    Returns:
        dict: 包含降维结果的中间结果对象，形式为：
              {
                  key: {
                      "bkg_reduced": torch.Tensor,  # bkg_acts 的降维结果
                      "act_reduced": torch.Tensor,  # act_with_lengths 的降维结果
                      "mds_r2": float              # MDS R² 值
                  },
                  ...
              }
    """
    results = {}

    # 遍历每个 key (0 到 31)
    for key in range(32):
        # 获取当前 key 对应的背景数据和 act 数据
        bkg_data = bkg_acts_tensor[key]  # 背景数据，形状为 (N, D)
        act_data = act_with_lengths_tensor[key]  # act 数据，形状为 (M, D)

        # 合并两个集合
        combined_data = torch.cat([bkg_data, act_data], dim=0)  # 合并为 (N + M, D)

        # 计算 pairwise 距离矩阵 (使用 minkowski_pairs)
        delta = minkowski_pairs(combined_data, sqform=False)  # 返回长格式的 pairwise 距离矩阵

        # 使用 mdscuda 的 MDS 类执行降维
        mds = MDS(n_dims=n_dims, verbosity=1)  # 创建 MDS 对象
        reduced_data = mds.fit(delta, calc_r2=True)  # 执行降维
        print(f"Key {key}: MDS R²: {mds.r2}")

        # 分离降维后的背景数据和 act 数据
        bkg_reduced = reduced_data[:len(bkg_data)]  # 前 N 行是 bkg 的降维结果
        act_reduced = reduced_data[len(bkg_data):]  # 后 M 行是 act 的降维结果

        # 保存降维结果和 R² 值
        results[key] = {
            "bkg_reduced": torch.tensor(bkg_reduced),  # 转换为 torch.Tensor
            "act_reduced": torch.tensor(act_reduced),  # 转换为 torch.Tensor
            "mds_r2": mds.r2
        }
    
    return results


bkg_prompts=json.load(open('/mnt/data/users/Lang_Gao/proj/Fairness_Mechanisms/alpaca_samples_lengthbalanced.json'))
qs=json.load(open('/mnt/data/users/Lang_Gao/proj/Fairness_Mechanisms/dev-v2.0.json','r'))
# query=random.choice(qs['data'])['paragraphs'][0]['qas'][0]['question']
query="1+1=?"
sequencial_activations=[]
# for query in tqdm(queries):
#     print(query)
sequencial_activations.append(get_subsequent_activations(model,tokenizer,query))
bkg=torch.load('/mnt/data/users/Lang_Gao/proj/Fairness_Mechanisms/tensors/bkg_acts_alpaca_tensor.pt')
def prepare_sequential_activations(sequencial_activations, num_layers=32):
    """
    将 sequencial_activations 转换为 compute_mds 所需的格式。

    Parameters:
        sequencial_activations (list): 长度为 10 的列表，每个元素是字典：
                                       {layer_id: list of torch.Tensor (780, 4096)}
        num_layers (int): 模型的层数（默认为 32 层）。

    Returns:
        list[torch.Tensor]: 长度为 num_layers 的列表，每个元素是 torch.Tensor，
                            形状为 (M, 4096)，表示该层所有 prompts 的 activations 合并结果。
        list: 每层的索引映射，用于之后还原原始结构。
    """
    # 初始化每层的激活列表
    act_with_lengths_tensor = [torch.empty((0, 4096)) for _ in range(num_layers)]
    layer_indices = [{} for _ in range(num_layers)]  # 保存每层的索引映射

    # 遍历每个 prompt
    for prompt_id, prompt_activations in enumerate(sequencial_activations):
        for layer_id, activations in prompt_activations.items():
            # 将当前 prompt 的 activations（list of torch.Tensor）拼接到对应层
            activations_tensor = torch.stack(activations, dim=0)  # (780, 4096)
            start_idx = act_with_lengths_tensor[layer_id].shape[0]  # 获取合并前的起始索引
            end_idx = start_idx + activations_tensor.shape[0]  # 计算结束索引
            act_with_lengths_tensor[layer_id] = torch.cat([act_with_lengths_tensor[layer_id], activations_tensor], dim=0)
            
            # 保存索引映射
            layer_indices[layer_id][prompt_id] = (start_idx, end_idx)

    return act_with_lengths_tensor, layer_indices


def restore_sequential_activations(reduced_activations, layer_indices, num_prompts=10):
    """
    将降维后的 activations 恢复为原始 `sequencial_activations` 的结构。

    Parameters:
        reduced_activations (list[torch.Tensor]): 长度为 32 的列表，每个元素是 torch.Tensor，
                                                  形状为 (M, 2)，表示降维后的 activations。
        layer_indices (list): 每层的索引映射，用于恢复原始结构。
        num_prompts (int): prompt 的数量（默认为 10）。

    Returns:
        list: 恢复后的 `sequencial_activations`，保持与输入时的嵌套结构一致。
    """
    # 初始化恢复后的结构
    restored_activations = [{} for _ in range(num_prompts)]

    # 遍历每层的降维结果
    for layer_id, reduced_layer_activations in enumerate(reduced_activations):
        for prompt_id, (start_idx, end_idx) in layer_indices[layer_id].items():
            # 从降维结果中提取当前 prompt 的 activations
            prompt_activations = reduced_layer_activations[start_idx:end_idx]
            # 保存到恢复后的结构中
            if layer_id not in restored_activations[prompt_id]:
                restored_activations[prompt_id][layer_id] = []
            restored_activations[prompt_id][layer_id] = prompt_activations

    return restored_activations


# 主流程

# Step 1: 准备 sequencial_activations 为 compute_mds 的输入格式
act_with_lengths_tensor, layer_indices = prepare_sequential_activations(sequencial_activations)

# Step 2: 调用 compute_mds 进行降维
results = compute_mds(bkg, act_with_lengths_tensor)

# Step 3: 恢复 sequencial_activations 的原始结构
reduced_activations = [results[key]["act_reduced"] for key in range(32)]  # 提取降维结果
restored_activations = restore_sequential_activations(reduced_activations, layer_indices)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_combined_results_with_arrows(results, restored_activations):
    """
    绘制背景激活（bkg）和 query 激活（restored_activations）的降维结果，并在 query 点之间添加箭头。
    添加一个深蓝色粗箭头从起始蓝色点指向最终的黑色点。

    Parameters:
        results (dict): `compute_mds` 函数的返回值，包含背景激活的降维结果。
        restored_activations (list): 降维后的 `sequencial_activations`，保持原始嵌套结构。
                                     每个元素是一个字典，表示一个 query 的降维结果：
                                     [{layer_id: torch.Tensor (780, 2)}, ...]
    """
    num_layers = len(results)  # 总共 32 层
    num_queries = len(restored_activations)  # Query 数量

    # 动态调整图像大小
    num_cols = 8  # 每行显示 8 个子图
    num_rows = (num_layers + num_cols - 1) // num_cols  # 根据层数计算行数
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    if num_rows == 1:
        axes = [axes]  # 转换为列表形式
    else:
        axes = axes.ravel()  # 将多维数组展平成一维列表

    fig.suptitle('MDS Visualization: Background + Query Activations', fontsize=20)  # 全局标题

    # 遍历每个层的背景激活结果
    for layer_id, (key, data) in enumerate(results.items()):
        ax = axes[layer_id]  # 获取当前层的子图

        # 提取背景激活的降维结果
        bkg_reduced = data["bkg_reduced"]  # 形状为 (N, 2)
        mds_r2 = data["mds_r2"]  # R² 值

        # 将背景数据从 torch.Tensor 转换为 numpy 数组
        bkg_reduced = bkg_reduced.cpu().numpy() if isinstance(bkg_reduced, torch.Tensor) else bkg_reduced

        # 计算每个维度的 5% 和 95% 分位数
        q_low = np.percentile(bkg_reduced, 1, axis=0)
        q_high = np.percentile(bkg_reduced, 85, axis=0)

        # 筛选数据点，只保留聚集在 90% 分位数范围内的点
        mask = np.all((bkg_reduced >= q_low) & (bkg_reduced <= q_high), axis=1)
        filtered_data = bkg_reduced[mask]

        # 生成绿色到红色的颜色映射
        num_points = filtered_data.shape[0]
        cmap = mcolors.LinearSegmentedColormap.from_list("green_to_black", ["green", "red"])
        colors = [cmap(idx / (num_points - 1)) for idx in range(num_points)]

        # 绘制背景激活分布
        ax.scatter(filtered_data[:, 0], filtered_data[:, 1], c=colors, alpha=0.5, label='Background')

        # 叠加 query 激活逐点绘制并添加箭头
        for query_id, query_data in enumerate(restored_activations):
            if layer_id not in query_data:  # 如果该 query 没有对应层的数据，跳过
                continue

            # 提取该 query 在当前层的降维结果
            query_reduced = query_data[layer_id]
            query_reduced = query_reduced.cpu().numpy() if isinstance(query_reduced, torch.Tensor) else query_reduced

            # 生成蓝到黑的颜色映射，行下标越小越蓝，越大越黑
            num_query_points = query_reduced.shape[0]
            cmap_query = mcolors.LinearSegmentedColormap.from_list("blue_to_black", ["blue", "black"])
            query_colors = [
                cmap_query(idx / (num_query_points - 1)) for idx in range(num_query_points)
            ]

            # 绘制 query 激活散点
            ax.scatter(
                query_reduced[:, 0],
                query_reduced[:, 1],
                c=query_colors,
                alpha=0.8,
                label=f'Query {query_id}' if layer_id == 0 else None  # 图例仅在第一个子图显示
            )

            # 添加箭头（相邻的点之间）
            for i in range(len(query_reduced) - 1):
                start = query_reduced[i]
                end = query_reduced[i + 1]
                ax.annotate(
                    '',  # 空文本
                    xy=end,  # 箭头指向
                    xytext=start,  # 箭头起点
                    arrowprops=dict(
                        arrowstyle="->",
                        color="lightblue",
                        lw=1.5,
                        alpha=0.7
                    )
                )

            # 添加深蓝色的粗箭头：从第一个点指向最后一个点
            start_point = query_reduced[0]
            end_point = query_reduced[-1]
            ax.annotate(
                '',  # 空文本
                xy=end_point,  # 箭头指向
                xytext=start_point,  # 箭头起点
                arrowprops=dict(
                    arrowstyle="->",
                    color="darkblue",
                    lw=3.5,  # 粗箭头线宽
                    alpha=0.9
                )
            )

        # 设置子图标题
        ax.set_title(f"Layer {key} (R²={mds_r2:.2f})", fontsize=10)

        # 隐藏坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])

    # 删除多余的子图（如果层数少于网格的总子图数）
    for idx in range(num_layers, len(axes)):
        fig.delaxes(axes[idx])  # 删除未使用的子图

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出全局标题的空间
    plt.savefig('/mnt/data/users/Lang_Gao/proj/Fairness_Mechanisms/visualizations/combined_results_with_arrows.png')
    plt.show()
    
import torch
import matplotlib.pyplot as plt
def plot_activation_distances(sequencial_activations):
    """
    计算并绘制每个 token 相对于最后一个 token 的距离变化图。

    Parameters:
        sequencial_activations (list): 每个 token 的潜在特征向量列表。
                                       列表中的每个元素是一个字典，表示每一层的特征向量：
                                       [{0: tensor(N, D), 1: tensor(N, D), ...},  # 第一个 token
                                        {0: tensor(N, D), 1: tensor(N, D), ...},  # 第二个 token
                                        ...
                                       ]
                                       其中，N 是层数，D 是特征向量维度。
    """
    num_tokens = len(sequencial_activations)  # Token 的数量
    num_layers = len(sequencial_activations[0])  # 假设每个 token 的层数相同

    # 动态调整图像大小
    num_cols = 8  # 每行显示 8 个子图
    num_rows = (num_layers + num_cols - 1) // num_cols  # 根据层数计算行数
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    if num_rows == 1:
        axes = [axes]  # 转换为列表形式
    else:
        axes = axes.ravel()  # 将多维数组展平成一维列表

    fig.suptitle('Distances Between Tokens and Last Token Across Layers', fontsize=20)  # 全局标题

    # 遍历每一层
    for layer_id in range(num_layers):
        ax = axes[layer_id]  # 获取当前层的子图

        # 提取所有 token 在当前层的特征向量，形状为 (num_tokens, D)
        layer_activations = torch.stack([token_data[layer_id] for token_data in sequencial_activations])  # (num_tokens, D)

        # 提取最后一个 token 在当前层的特征向量
        last_token_vector = layer_activations[-1]  # (D,)

        # 计算每个 token 相对于最后一个 token 的距离
        distances = torch.norm(layer_activations - last_token_vector, dim=1)  # (num_tokens,)

        # 绘制距离变化曲线
        ax.plot(
            range(num_tokens),  # X 轴为 token 的索引
            distances.cpu().numpy(),  # Y 轴为距离
            label=f'Layer {layer_id}',
            alpha=0.8
        )

        # 设置子图标题
        ax.set_title(f"Layer {layer_id}", fontsize=10)

        # 设置坐标轴标签
        ax.set_xlabel("Token Index", fontsize=8)
        ax.set_ylabel("Distance", fontsize=8)

    # 删除多余的子图（如果层数少于网格的总子图数）
    for idx in range(num_layers, len(axes)):
        fig.delaxes(axes[idx])  # 删除未使用的子图

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出全局标题的空间
    plt.savefig('/mnt/data/users/Lang_Gao/proj/Fairness_Mechanisms/visualizations/token_distances_to_last_token.png')
    plt.show()

def plot_pairwise_token_distances(sequencial_activations):
    """
    计算并绘制每个 token（第 0 个除外）相对于其之前一个 token 的距离变化图。

    Parameters:
        sequencial_activations (list): 每个 token 的潜在特征向量列表。
                                       列表中的每个元素是一个字典，表示每一层的特征向量：
                                       [{0: tensor(N, D), 1: tensor(N, D), ...},  # 第一个 token
                                        {0: tensor(N, D), 1: tensor(N, D), ...},  # 第二个 token
                                        ...
                                       ]
                                       其中，N 是层数，D 是特征向量维度。
    """
    num_tokens = len(sequencial_activations)  # Token 的数量
    num_layers = len(sequencial_activations[0])  # 假设每个 token 的层数相同

    # 动态调整图像大小
    num_cols = 8  # 每行显示 8 个子图
    num_rows = (num_layers + num_cols - 1) // num_cols  # 根据层数计算行数
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    if num_rows == 1:
        axes = [axes]  # 转换为列表形式
    else:
        axes = axes.ravel()  # 将多维数组展平成一维列表

    fig.suptitle('Pairwise Token Distances Across Layers', fontsize=20)  # 全局标题

    # 遍历每一层
    for layer_id in range(num_layers):
        ax = axes[layer_id]  # 获取当前层的子图

        # 提取所有 token 在当前层的特征向量，形状为 (num_tokens, D)
        layer_activations = torch.stack([token_data[layer_id] for token_data in sequencial_activations])  # (num_tokens, D)

        # 计算相邻 token 之间的距离
        distances = torch.norm(layer_activations[1:] - layer_activations[:-1], dim=1)  # (num_tokens - 1,)

        # 绘制距离变化曲线
        ax.plot(
            range(1, num_tokens),  # X 轴为 token 的索引（从第 1 个 token 开始）
            distances.cpu().numpy(),  # Y 轴为距离
            label=f'Layer {layer_id}',
            alpha=0.8
        )

        # 设置子图标题
        ax.set_title(f"Layer {layer_id}", fontsize=10)

        # 设置坐标轴标签
        ax.set_xlabel("Token Index", fontsize=8)
        ax.set_ylabel("Distance", fontsize=8)

    # 删除多余的子图（如果层数少于网格的总子图数）
    for idx in range(num_layers, len(axes)):
        fig.delaxes(axes[idx])  # 删除未使用的子图

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出全局标题的空间
    plt.savefig('/mnt/data/users/Lang_Gao/proj/Fairness_Mechanisms/visualizations/token_pairwise_distances.png')
    plt.show()

    
plot_combined_results_with_arrows(results, restored_activations)
plot_activation_distances(sequencial_activations[0])
plot_pairwise_token_distances(sequencial_activations[0])