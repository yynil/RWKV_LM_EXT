input_text = '粒子滤波是基于递推的Monte Carlo仿真方法的总称,原则上可用于任意非线性、非高斯随机系统的状态估计.但粒子的退化现象是粒子滤波器的最大缺陷,减轻这一现象影响的方法之一就是对粒子进行重采样.阐述了几种重要的重采样算法及其改进策略,并对其进行总结与展望.'
task = {"领域": "算法与计算机科学", "专家": "数据科学家/算法工程师", "任务": ["理解粒子滤波的基本原理和应用范围", "分析粒子滤波器的退化问题及其影响", "研究并实现几种重要的重采样算法", "评估不同重采样策略的效果和效率", "探索和提出减轻粒子退化现象的新策略"]}
input_data = {"input": input_text, "task": task}
instruction = '据input中的领域和任务，协助用户识别input文本中存在的实体类型。 实体类型必须与用户任务相关。 避免使用诸如“其他”或“未知”的通用实体类型。 非常重要的是：不要生成冗余或重叠的实体类型。用JSON格式输出。'
output = {'entity_types': ['算法与计算机科学', '状态估计', '非线性系统', '非高斯随机系统', '粒子滤波器', '重采样算法', '改进策略']}

import json
final_data = {
    "input": json.dumps(input_data, ensure_ascii=False),
    "instruction": instruction,
    "output": json.dumps(output, ensure_ascii=False)
}
final_str = json.dumps(final_data, ensure_ascii=False)
print(final_str)