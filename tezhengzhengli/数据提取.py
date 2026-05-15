import os
import re

# 1. 定义你存放提取文件的目标文件夹路径
# 请确认这和你刚才提取存放的路径一致
target_dir = r"C:\Users\云瑾\Desktop\data\提取的SIEEG_main数据"

print(f"开始扫描并重命名文件夹：{target_dir}")
count = 0

# 2. 设定匹配模板：提取字符串结尾的 "SIEEG (数字)_main.csv"
pattern = re.compile(r"(SIEEG \(\d+\)_main\.csv)$")

for filename in os.listdir(target_dir):
    # 查找文件名中符合条件的部分
    match = pattern.search(filename)

    if match:
        # match.group(1) 就是提取出来的干净名字，比如 "SIEEG (1)_main.csv"
        clean_name = match.group(1)

        # 如果文件本身就已经叫这个名字了，就跳过
        if filename == clean_name:
            continue

        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, clean_name)

        try:
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"✅ 成功重命名: {filename}  ->  {clean_name}")
            count += 1
        except FileExistsError:
            # 增加一个保险：万一有重名的数字，防止覆盖报错
            print(f"⚠️ 警告: 无法重命名 {filename}，因为目标文件夹中已经存在 {clean_name}！")

# 3. 完成提示
print("-" * 40)
print(f"🎉 批量重命名完成！一共处理了 {count} 个文件。")