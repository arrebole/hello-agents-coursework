/**
 * TerminalTool - 命令行工具
 *
 * 为 Agent 提供受限的命令行执行能力，支持常见文件系统操作、文本处理、
 * 目录导航和输出限制。执行时只允许白名单首命令，并把工作目录限制在
 * 指定 workspace 内。
 */

import { mkdirSync, statSync } from "fs";
import { platform } from "os";
import { isAbsolute, resolve } from "path";
import { spawnSync } from "child_process";

import { Tool, ToolParameter } from "../../../7.hello-agents/tools/base";

type TerminalOsType = "auto" | "windows" | "linux" | "mac";
type ResolvedOsType = Exclude<TerminalOsType, "auto">;

interface TerminalToolOptions {
    workspace?: string;
    timeout?: number;
    maxOutputSize?: number;
    allowCd?: boolean;
    osType?: TerminalOsType;
}

interface TerminalToolParams {
    command?: string;
}

interface ParsedCommand {
    command: string;
    args: string[];
}

const DEFAULT_MAX_OUTPUT_SIZE = 10 * 1024 * 1024;

const ALLOWED_COMMANDS = new Set([
    // 文件列表与信息
    "ls", "dir", "tree",
    // 文件内容查看
    "cat", "type", "head", "tail", "less", "more",
    // 文件搜索
    "find", "where", "grep", "egrep", "fgrep", "findstr",
    // 文本处理
    "wc", "sort", "uniq", "cut", "awk", "sed",
    // 目录操作
    "pwd", "cd",
    // 文件信息
    "file", "stat", "du", "df",
    // 其他
    "echo", "which", "whereis",
    // 代码执行
    "python", "python3", "node", "bash", "sh", "powershell", "cmd",
]);

const SHELL_CONTROL_TOKENS = new Set([";", "&&", "||", "|", ">", ">>", "<", "<<"]);
const PATH_ARGUMENT_COMMANDS = new Set([
    "ls", "dir", "tree", "cat", "type", "head", "tail", "less", "more",
    "find", "grep", "egrep", "fgrep", "findstr", "wc", "sort", "uniq",
    "cut", "awk", "sed", "file", "stat", "du",
]);

function detectOs(): ResolvedOsType {
    const system = platform().toLowerCase();
    if (system === "win32") return "windows";
    if (system === "darwin") return "mac";
    return "linux";
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function normalizeTimeout(value: number): number {
    if (!Number.isFinite(value) || value <= 0) return 30;
    return Math.floor(value);
}

function normalizeMaxOutputSize(value: number): number {
    if (!Number.isFinite(value) || value <= 0) return DEFAULT_MAX_OUTPUT_SIZE;
    return Math.floor(value);
}

function isOptionToken(token: string): boolean {
    return token.startsWith("-") || token.startsWith("/");
}

function hasGlob(token: string): boolean {
    return /[*?[\]]/.test(token);
}

function isLikelyPathToken(token: string): boolean {
    if (!token || isOptionToken(token)) return false;
    if (token.includes("=")) return false;
    return token === "."
        || token === ".."
        || token.startsWith("./")
        || token.startsWith("../")
        || token.startsWith("~/")
        || isAbsolute(token)
        || token.includes("/")
        || token.includes("\\")
        || /^[A-Za-z]:[\\/]/.test(token);
}

function normalizePathForDisplay(pathValue: string): string {
    return pathValue.replace(/\\/g, "/");
}

/**
 * 最小 shell-like 分词器。
 *
 * 支持单引号、双引号和反斜杠转义；不执行变量展开、glob 展开或命令替换。
 */
function splitCommand(command: string): string[] {
    const tokens: string[] = [];
    let current = "";
    let quote: "'" | "\"" | null = null;
    let escaping = false;

    for (const char of command) {
        if (escaping) {
            current += char;
            escaping = false;
            continue;
        }

        if (char === "\\") {
            escaping = true;
            continue;
        }

        if (quote) {
            if (char === quote) {
                quote = null;
            } else {
                current += char;
            }
            continue;
        }

        if (char === "'" || char === "\"") {
            quote = char;
            continue;
        }

        if (/\s/.test(char)) {
            if (current.length > 0) {
                tokens.push(current);
                current = "";
            }
            continue;
        }

        current += char;
    }

    if (escaping) current += "\\";
    if (quote) throw new Error("引号未闭合");
    if (current.length > 0) tokens.push(current);

    return tokens;
}

function parseCommand(command: string): ParsedCommand {
    const parts = splitCommand(command);
    if (parts.length === 0) throw new Error("命令不能为空");

    const [baseCommand, ...args] = parts;
    return { command: baseCommand, args };
}

/**
 * 跨平台命令行工具。
 *
 * 默认白名单覆盖文件查看、搜索、文本处理和常见解释器命令。解释器命令仍以
 * `spawnSync` 的非 shell 模式执行，避免 shell 控制符绕过首命令白名单。
 */
export class TerminalTool extends Tool {
    static readonly ALLOWED_COMMANDS = ALLOWED_COMMANDS;

    private readonly workspace: string;
    private readonly timeout: number;
    private readonly maxOutputSize: number;
    private readonly allowCd: boolean;
    private readonly osType: ResolvedOsType;
    private currentDir: string;

    constructor(workspace?: string, timeout?: number, maxOutputSize?: number, allowCd?: boolean, osType?: TerminalOsType);
    constructor(options?: TerminalToolOptions);
    constructor(
        workspaceOrOptions: string | TerminalToolOptions = ".",
        timeout = 30,
        maxOutputSize = DEFAULT_MAX_OUTPUT_SIZE,
        allowCd = true,
        osType: TerminalOsType = "auto",
    ) {
        const options = typeof workspaceOrOptions === "string"
            ? { workspace: workspaceOrOptions, timeout, maxOutputSize, allowCd, osType }
            : workspaceOrOptions;

        super(
            "terminal",
            "跨平台命令行工具 - 执行安全的文件系统、文本处理和代码执行命令（支持Windows/Linux/Mac）",
        );

        this.workspace = resolve(options.workspace ?? ".");
        this.timeout = normalizeTimeout(options.timeout ?? 30);
        this.maxOutputSize = normalizeMaxOutputSize(options.maxOutputSize ?? DEFAULT_MAX_OUTPUT_SIZE);
        this.allowCd = options.allowCd ?? true;
        this.osType = options.osType && options.osType !== "auto"
            ? options.osType
            : detectOs();
        this.currentDir = this.workspace;

        mkdirSync(this.workspace, { recursive: true });
    }

    run(parameters: Record<string, unknown>): string {
        if (!this.validateParameters(parameters)) {
            return "❌ 参数验证失败";
        }

        const command = String((parameters as TerminalToolParams).command ?? "").trim();
        if (!command) {
            return "❌ 命令不能为空";
        }

        let parsed: ParsedCommand;
        try {
            parsed = parseCommand(command);
        } catch (error) {
            return `❌ 命令解析失败: ${errorMessage(error)}`;
        }

        if (!ALLOWED_COMMANDS.has(parsed.command)) {
            return `❌ 不允许的命令: ${parsed.command}\n允许的命令: ${[...ALLOWED_COMMANDS].sort().join(", ")}`;
        }

        const unsafeToken = [parsed.command, ...parsed.args].find((token) => SHELL_CONTROL_TOKENS.has(token));
        if (unsafeToken) {
            return `❌ 不允许使用 shell 控制符: ${unsafeToken}`;
        }

        if (parsed.command === "cd") {
            return this.handleCd(parsed.args);
        }

        const pathValidationError = this.validatePathArguments(parsed);
        if (pathValidationError) return pathValidationError;

        return this.executeCommand(parsed);
    }

    getParameters(): ToolParameter[] {
        return [
            new ToolParameter({
                name: "command",
                type: "string",
                description: `要执行的命令（白名单: ${[...ALLOWED_COMMANDS].sort().slice(0, 10).join(", ")}...）。示例: 'ls -la', 'cat file.txt', 'grep pattern src/file.ts', 'head -n 20 data.csv'`,
                required: true,
            }),
        ];
    }

    getCurrentDir(): string {
        return this.currentDir;
    }

    get_current_dir(): string {
        return this.getCurrentDir();
    }

    resetDir(): void {
        this.currentDir = this.workspace;
    }

    reset_dir(): void {
        this.resetDir();
    }

    getOsType(): ResolvedOsType {
        return this.osType;
    }

    get_os_type(): ResolvedOsType {
        return this.getOsType();
    }

    private handleCd(args: string[]): string {
        if (!this.allowCd) {
            return "❌ cd 命令已禁用";
        }

        if (args.length === 0) {
            return `当前目录: ${this.currentDir}`;
        }

        const targetDir = args[0];
        const newDir = this.resolveWorkspacePath(targetDir === "~" ? "." : targetDir);

        if (!this.isInsideWorkspace(newDir)) {
            return `❌ 不允许访问工作目录外的路径: ${newDir}`;
        }

        try {
            const stats = statSync(newDir);
            if (!stats.isDirectory()) {
                return `❌ 不是目录: ${newDir}`;
            }
        } catch {
            return `❌ 目录不存在: ${newDir}`;
        }

        this.currentDir = newDir;
        return `✅ 切换到目录: ${this.currentDir}`;
    }

    private executeCommand(parsed: ParsedCommand): string {
        try {
            const result = spawnSync(parsed.command, parsed.args, {
                cwd: this.currentDir,
                env: process.env,
                encoding: "utf-8",
                timeout: this.timeout * 1000,
                maxBuffer: this.maxOutputSize,
                shell: false,
            });

            if (result.error) {
                if ((result.error as NodeJS.ErrnoException).code === "ETIMEDOUT") {
                    return `❌ 命令执行超时（超过 ${this.timeout} 秒）`;
                }
                return `❌ 命令执行失败: ${result.error.message}`;
            }

            let output = result.stdout ?? "";
            if (result.stderr) {
                output += `${output ? "\n" : ""}[stderr]\n${result.stderr}`;
            }

            if (output.length > this.maxOutputSize) {
                output = output.slice(0, this.maxOutputSize);
                output += `\n\n⚠️ 输出被截断（超过 ${this.maxOutputSize} 字符）`;
            }

            if ((result.status ?? 0) !== 0) {
                output = `⚠️ 命令返回码: ${result.status}\n\n${output}`;
            }

            return output || "✅ 命令执行成功（无输出）";
        } catch (error) {
            return `❌ 命令执行失败: ${errorMessage(error)}`;
        }
    }

    private validatePathArguments(parsed: ParsedCommand): string | null {
        if (!PATH_ARGUMENT_COMMANDS.has(parsed.command)) return null;

        for (const arg of parsed.args) {
            if (arg === "--") continue;
            if (isOptionToken(arg)) continue;
            if (hasGlob(arg)) continue;
            if (!isLikelyPathToken(arg)) continue;

            const resolvedPath = this.resolveWorkspacePath(arg);
            if (!this.isInsideWorkspace(resolvedPath)) {
                return `❌ 不允许访问工作目录外的路径: ${resolvedPath}`;
            }
        }

        return null;
    }

    private resolveWorkspacePath(pathValue: string): string {
        if (pathValue === "~") return this.workspace;
        if (pathValue.startsWith("~/")) {
            return resolve(this.workspace, pathValue.slice(2));
        }
        if (isAbsolute(pathValue) || /^[A-Za-z]:[\\/]/.test(pathValue)) {
            return resolve(pathValue);
        }
        return resolve(this.currentDir, pathValue);
    }

    private isInsideWorkspace(pathValue: string): boolean {
        const normalizedWorkspace = normalizePathForDisplay(this.workspace);
        const normalizedPath = normalizePathForDisplay(resolve(pathValue));
        return normalizedPath === normalizedWorkspace
            || normalizedPath.startsWith(`${normalizedWorkspace}/`);
    }
}
