import torch
import os
import gc
import warnings
warnings.filterwarnings("ignore")

# 设置路径
input_dir = '/mnt/data/users/Lang_Gao/proj/att_activations/random_400/llama2'  # 输入pth文件的目录
output_dir = '/mnt/data/users/Lang_Gao/proj/att_activations/random_400/llama2/layers'  # 输出tensor的目录
block_count = 32  # 假设有32个block

# 创建32个子文件夹，每个子文件夹对应一个block
for block_num in range(0, block_count ):
    block_dir = os.path.join(output_dir, f'block_{block_num}')
    try:
        os.makedirs(block_dir, exist_ok=True)
        print(f"Directory created or already exists: {block_dir}")
    except Exception as e:
        print(f"Failed to create directory {block_dir}: {e}")

# 获取所有的pth文件
pth_files = [f for f in os.listdir(input_dir) if f.endswith('.pth')]

# 遍历每个pth文件
for pth_file in pth_files:
    print(f'Processing {pth_file}...')
    
    # 加载该pth文件
    pth_path = os.path.join(input_dir, pth_file)
    data_list = torch.load(pth_path)
    
    # 创建32个空列表来存储每个block的tensor
    block_tensors = {f'block_{i}': [] for i in range(0, block_count)}

    # 遍历文件中的每个样本
    for sample in data_list:
        label = sample['label']
        
        # 遍历每个block的激活值
        for block_num, activation in sample['activations'].items():
            block_name = f'block_{block_num.split("_")[-1]}'  # 获取block名称
            
            # 拼接label和该block的activation
            label_activation = torch.cat([torch.tensor([label]), activation], dim=0)
            
            # 将拼接后的tensor添加到对应的block列表中
            block_tensors[block_name].append(label_activation)

    # 将每个block的激活值保存到对应的子文件夹中
    for block_name, tensor_list in block_tensors.items():
        if tensor_list:
            # 将list转为tensor
            block_tensor = torch.stack(tensor_list)
            
            # 保存tensor到对应的子文件夹中，保存为.pt格式以节省空间
            output_path = os.path.join(output_dir, block_name, f'{pth_file.replace(".pth", "")}.pt')
            try:
                torch.save(block_tensor, output_path)
                print(f'Saved {block_name} activations to {output_path}')
            except Exception as e:
                print(f"Failed to save tensor for {block_name}: {e}")
    
    # 手动清理内存
    del data_list, block_tensors
    gc.collect()

# 第二阶段：拼接每个block的tensor并保存
print("Starting concatenation process...")

for block_num in range(0, block_count):
    block_name = f'block_{block_num}'
    block_dir = os.path.join(output_dir, block_name)
    
    # 获取该block目录下的所有tensor文件
    tensor_files = [f for f in os.listdir(block_dir) if f.endswith('.pt')]
    
    # 创建一个空列表来存储所有tensor
    all_tensors = []
    
    # 依次加载并拼接
    for tensor_file in tensor_files:
        tensor_path = os.path.join(block_dir, tensor_file)
        block_tensor = torch.load(tensor_path)
        all_tensors.append(block_tensor)
    
    # 将所有小tensor拼接为一个大tensor
    if all_tensors:
        concatenated_tensor = torch.cat(all_tensors, dim=0)
        
        # 保存拼接后的tensor
        final_output_path = os.path.join(output_dir, f'{block_name}_final.pt')
        try:
            torch.save(concatenated_tensor, final_output_path)
            print(f'Saved concatenated tensor for {block_name} to {final_output_path}')
        except Exception as e:
            print(f"Failed to save concatenated tensor for {block_name}: {e}")
    
    # 清理内存
    del all_tensors
    gc.collect()

print("All blocks processed and saved.")