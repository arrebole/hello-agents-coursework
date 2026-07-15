/**
 * HelloAgents 框架异常的统一根类型。
 *
 * 业务层可以通过 `instanceof HelloAgentsException` 将框架主动抛出的已知错误
 * 与原生异常或第三方 SDK 错误区分开；需要进一步判断故障领域时，再匹配下面的
 * LLM、Agent、配置或工具异常。
 *
 * 该基类目前只统一错误类型和消息，不会自动包装底层异常，也不保存 `cause`、
 * 错误码或上下文数据。调用方若需要保留这些信息，应在抛出异常的位置显式处理。
 */
export class HelloAgentsException extends Error {
    constructor(message?: string) {
        super(message);

        // Error 默认使用 "Error" 作为名称；显式设置后，日志和堆栈能稳定展示框架异常类型。
        this.name = 'HelloAgentsException';

        // 在部分转译目标中，继承内置 Error 后实例的原型链可能不完整。
        // 使用 new.target 可同时兼容基类和所有子类，保证 instanceof 判断结果可靠。
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

/**
 * 表示大语言模型领域的失败，例如模型请求、响应处理或服务提供方交互异常。
 *
 * 这是用于分类捕获的标记类型，本身不负责重试、降级或将 SDK 异常自动转换为该类型。
 */
export class LLMException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);

        // 覆盖基类名称，使日志能够直接反映具体的异常领域。
        this.name = 'LLMException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

/**
 * 表示 Agent 生命周期、任务编排或执行过程中的框架级失败。
 *
 * 工具自身的调用失败应使用 ToolException；模型服务交互失败应使用 LLMException，
 * 以便上层针对不同故障来源采取不同的恢复策略。
 */
export class AgentException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'AgentException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

/**
 * 表示配置缺失、配置值非法或配置组合不兼容等问题。
 *
 * 该类型适合在配置解析或启动校验阶段抛出；运行期间由外部服务返回的失败不应仅因
 * 与配置值有关就归入此类，而应优先使用对应运行领域的异常。
 */
export class ConfigException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'ConfigException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

/**
 * 表示工具注册、参数校验或工具执行过程中的失败。
 *
 * 此类型描述的是 Agent 所调用工具的边界错误，不用于表示 Agent 编排逻辑本身的异常。
 * 与其他领域子类一样，它只提供异常分类，不附带工具名、参数或执行结果等结构化信息。
 */
export class ToolException extends HelloAgentsException {
    constructor(message?: string) {
        super(message);
        this.name = 'ToolException';
        Object.setPrototypeOf(this, new.target.prototype);
    }
}
