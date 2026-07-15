// HelloAgents基础异常类
export class HelloAgentsException extends Error {
    constructor(message?: string) {
        super(message);
        this.name = 'HelloAgentsException';
        // 修复原型链
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

// LLM相关异常
export class LLMException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'LLMException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

// Agent相关异常
export class AgentException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'AgentException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

// 配置相关异常
export class ConfigException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'ConfigException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

// 工具相关异常
export class ToolException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'ToolException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}