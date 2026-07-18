import { MemoryTool } from './memory_tool';

const mempryTool = new MemoryTool();

// 1. 工作记忆 - 临时信息，容量有限
mempryTool.execute("add",
    "用户刚才问了关于Python函数的问题",
    "working",
    0.6
)

// # 2. 情景记忆 - 具体事件和经历
mempryTool.execute("add",
    "2024年3月15日，用户张三完成了第一个Python项目",
    "episodic",
    0.8,
    "milestone",
    "在线学习平台"
)

// # 3. 语义记忆 - 抽象知识和概念
mempryTool.execute("add",
    "Python是一种解释型、面向对象的编程语言",
   "semantic",
    0.9,
    "factual"
)

// # 4. 感知记忆 - 多模态信息
mempryTool.execute("add",
    "用户上传了一张Python代码截图，包含函数定义",
    "perceptual",
    0.7,
    "image",
    "./uploads/code_screenshot.png"
)

const result1 = mempryTool.execute(
    "search", 
    "Python编程", 
    5
)
console.log(result1);

// 指定记忆类型搜索
const result2 = mempryTool.execute("search",
    "学习进度",
    "episodic",
    3
)
console.log(result2);

// 多类型搜索
const result3 = mempryTool.execute("search",
    "函数定义",
    ["semantic", "episodic"],
    0.5
)
console.log(result3);