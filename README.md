# PHY_Master_v2 
# 使用说明
1. 【第一次使用】配置环境 （1）conda create -n phy python=3.10 （2）pip install -r requirements.txt
2. 启动环境 conda activate phy
3. 在instructions文件夹里新建 instructions/ <task_name>.txt
4. 如有有输入附件，请放在input文件夹内，并且在instruction txt文件中简要介绍一下，给出相对路径
5. 在config.yaml里将query_file路径改成 instructions/<task_name>.txt
6. 运行程序 python run.py

参数调节：
1. 知识库参数：
  1. enable=true即开启，将在迭代求解任务前检索相关文件，解析并提取定量与定性知识
  2. Write local to global：将本轮检索&解析到的知识加入长期知识库，以供检索增强生成（RAG）
  3. Global：enable=true则可以在长期知识库中检索，否则只在针对本任务构建的local知识库中检索

2. Monte Carlo树搜索参数
  1. parallel_process：并行扩展的节点数，确保 > max{draft_expension, revise_expension}即可保证最大效率
  2. max_rounds：MCTS迭代轮数上限，推荐在10～20之间
  3. draft_expension：针对新子任务起草新方法，建议设为2～4，是控制探索广度的主要参数
  4. revise_expension：针对有误的节点进行修改

# Agent输出内容格式
Agent输出为一个文件夹，路径为outputs/<task_name>，文件夹中包含：
- contract.json：经clarification后的结构化任务
- summary.json：如成功执行任务，对best trajectory的记录
- visualization.html：可视化的树搜索结构图，简明清楚地给出树结构与每个节点的基本信息，请从下载后在浏览器打开查看
每个node输出的所有内容为一个子文件夹，名称为depthX_NodeY，其中Y为节点的全局编号
目前，agent引入了子任务划分与动态schedule机制，所以每个node处理的是原任务划分后的一个子任务
每个子文件夹中包含
- .md文件：完整记录supervisor agent（负责schedule+critic）与theoretician（负责理论建模与代码计算）的交互流程，相对冗杂，内容在html中被整理得更清楚
- 其他输出文件：可能包含图片，csv数据表格等，会在md中有介绍
