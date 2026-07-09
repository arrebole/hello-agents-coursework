

// 注册和管理工具
export class ToolExecutor {
    private tools: Map<string, { description: string, func: any }>;

    public constructor() {
        this.tools = new Map();
    }

    public registerTool(name: string, description: string, func: any) {
        if (this.tools.has(name)) {
            console.log(`警告:工具 '${name}' 已存在，将被覆盖。`)
        }

        this.tools.set(name, { description, func });
    }

    public getTool(name: string): any {
        return this.tools.get(name)?.func;
    }

    public getAvailableTools() {
        const tools = []
        for (const i of this.tools.keys()) {
            tools.push(`- ${i}: ${this.tools.get(i)?.description}`)
        }
        return tools.join("\n");
    }
}