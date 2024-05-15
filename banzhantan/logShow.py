import re
import matplotlib.pyplot as plt


def plot_accuracy_from_log(file_path):
    # 读取日志文件的内容
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.readlines()

    # 使用正则表达式查找包含 'accuracy' 值的行
    accuracy_pattern = r'accuracy: (.*)'
    accuracy_matches = re.findall(accuracy_pattern, ' '.join(contents))

    # 将所有找到的准确率匹配项转换为浮点数
    accuracy_data = [float(num.strip()) for num in accuracy_matches]
    return  accuracy_data


# 示例如何使用此函数
file_path = '../attack_0_label0AsAttack.log'
filw_path_2 = "../newMethod.log"
file_base = "../no_attack.log"
acc_1 = plot_accuracy_from_log(file_path)
acc_2 = plot_accuracy_from_log(filw_path_2)
acc_base = plot_accuracy_from_log(file_base)
# 绘制准确率数据
plt.figure(figsize=(10, 5))
plt.plot(acc_1, marker='o', color='red')
plt.plot(acc_2, marker='o', color='blue')
plt.plot(acc_base, marker='*', color='black',label='base')
plt.title('Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
