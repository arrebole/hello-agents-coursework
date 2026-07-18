import { MemoryTool } from './memory_tool';

const memoryTool = new MemoryTool();

// 1. 工作记忆 - 临时信息，容量有限
memoryTool.run({
    action: "add",
    content: "用户刚才问了关于Python函数的问题",
    memoryType: "working",
    importance: 0.6,
});

// 2. 情景记忆 - 具体事件和经历
memoryTool.run({
    action: "add",
    content: "2024年3月15日，用户张三完成了第一个Python项目",
    memoryType: "episodic",
    importance: 0.8,
});

// 3. 语义记忆 - 抽象知识和概念
memoryTool.run({
    action: "add",
    content: "Python是一种解释型、面向对象的编程语言",
    memoryType: "semantic",
    importance: 0.9,
});

// 4. 感知记忆 - 多模态信息
memoryTool.run({
    action: "add",
    content: "用户上传了一张Python代码截图，包含函数定义",
    memoryType: "perceptual",
    importance: 0.7,
    modality: "image",
    filePath: "./uploads/code_screenshot.png",
});

const result1 = memoryTool.run({
    action: "search",
    query: "Python编程",
    limit: 5,
});
console.log(result1);

// 指定记忆类型搜索
const result2 = memoryTool.run({
    action: "search",
    query: "学习进度",
    memoryType: "episodic",
    limit: 3,
});
console.log(result2);

// 指定类型 + 最低重要性搜索
const result3 = memoryTool.run({
    action: "search",
    query: "函数定义",
    memoryType: "semantic",
    minImportance: 0.5,
});
console.log(result3);
