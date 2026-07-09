/**
 * 一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
 */
export class Memory {
    private records: Array<{ type: string; content: string }> = [];

    /**
     * 初始化一个空列表来存储所有记录。
     */
    constructor() {
        this.records = [];
    }

    /**
     * 向记忆中添加一条新记录。
     * 
     * @param recordType - 记录的类型 ('execution' 或 'reflection')
     * @param content - 记录的具体内容 (例如，生成的代码或反思的反馈)
     */
    addRecord(recordType: string, content: string): void {
        const record = { type: recordType, content: content };
        this.records.push(record);
        console.log(`📝 记忆已更新，新增一条 '${recordType}' 记录。`);
    }

    /**
     * 将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
     */
    getTrajectory(): string {
        const trajectoryParts: string[] = [];

        for (const record of this.records) {
            if (record.type === 'execution') {
                trajectoryParts.push(`--- 上一轮尝试 (代码) ---\n${record.content}`);
            } else if (record.type === 'reflection') {
                trajectoryParts.push(`--- 评审员反馈 ---\n${record.content}`);
            }
        }

        return trajectoryParts.join("\n\n");
    }

    /**
     * 获取最近一次的执行结果 (例如，最新生成的代码)。
     * 如果不存在，则返回 null。
     */
    getLastExecution(): string | null {
        // 从后向前遍历
        for (let i = this.records.length - 1; i >= 0; i--) {
            const record = this.records[i];
            if (record.type === 'execution') {
                return record.content;
            }
        }
        return null;
    }

    /**
     * 获取所有记录（可选辅助方法）
     */
    getAllRecords(): Array<{ type: string; content: string }> {
        return [...this.records];
    }

    /**
     * 清空记忆（可选辅助方法）
     */
    clear(): void {
        this.records = [];
        console.log("🗑️ 记忆已清空。");
    }

    /**
     * 获取记录数量（可选辅助方法）
     */
    getRecordCount(): number {
        return this.records.length;
    }
}