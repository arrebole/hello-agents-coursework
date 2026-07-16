/** 安全的数学表达式计算器，不使用 eval 或 Function 动态执行用户输入。 */

import { Tool, ToolParameter } from "../base";

type NumericKind = "integer" | "float";

/** 保存 Python 数值语义所需的最小类型信息，例如区分 4 与 4.0。 */
interface NumericValue {
  value: number;
  kind: NumericKind;
}

type MathFunction = (args: NumericValue[]) => NumericValue;

type Token =
  | { type: "number"; text: string }
  | { type: "identifier"; text: string }
  | { type: "operator"; text: "+" | "-" | "*" | "/" | "**" | "^" }
  | { type: "leftParenthesis" | "rightParenthesis" | "comma" | "eof"; text: string };

/**
 * 词法分析器只识别数字、名称和白名单运算符。
 * 属性访问、下标、字符串和任意 JavaScript 语法都会在这一层直接拒绝。
 */
class Lexer {
  private position = 0;
  private readonly source: string;

  constructor(source: string) {
    this.source = source;
  }

  tokenize(): Token[] {
    const tokens: Token[] = [];
    while (this.position < this.source.length) {
      const character = this.source[this.position];
      if (/\s/.test(character)) {
        this.position += 1;
        continue;
      }

      const remaining = this.source.slice(this.position);
      const number = remaining.match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
      if (number) {
        tokens.push({ type: "number", text: number[0] });
        this.position += number[0].length;
        continue;
      }

      const identifier = remaining.match(/^[A-Za-z_]\w*/);
      if (identifier) {
        tokens.push({ type: "identifier", text: identifier[0] });
        this.position += identifier[0].length;
        continue;
      }

      if (remaining.startsWith("**")) {
        tokens.push({ type: "operator", text: "**" });
        this.position += 2;
        continue;
      }

      if (character === "+" || character === "-" || character === "*" || character === "/" || character === "^") {
        tokens.push({ type: "operator", text: character });
      } else if (character === "(") {
        tokens.push({ type: "leftParenthesis", text: character });
      } else if (character === ")") {
        tokens.push({ type: "rightParenthesis", text: character });
      } else if (character === ",") {
        tokens.push({ type: "comma", text: character });
      } else {
        throw new SyntaxError(`不支持的字符: ${character}`);
      }
      this.position += 1;
    }

    tokens.push({ type: "eof", text: "" });
    return tokens;
  }
}

/**
 * 递归下降解析器同时完成求值。
 * 各层函数对应 Python 运算符优先级，尤其保留 `-2**2 === -4` 和幂运算右结合。
 */
class ExpressionParser {
  private position = 0;
  private readonly tokens: Token[];
  private readonly constants: Readonly<Record<string, NumericValue>>;
  private readonly functions: Readonly<Record<string, MathFunction>>;

  constructor(
    tokens: Token[],
    constants: Readonly<Record<string, NumericValue>>,
    functions: Readonly<Record<string, MathFunction>>,
  ) {
    this.tokens = tokens;
    this.constants = constants;
    this.functions = functions;
  }

  parse(): NumericValue {
    const result = this.parseXor();
    if (this.current().type !== "eof") {
      throw new SyntaxError(`意外的内容: ${this.current().text}`);
    }
    return result;
  }

  /** 位异或优先级低于加减法，与 Python 表达式规则一致。 */
  private parseXor(): NumericValue {
    let left = this.parseAddition();
    while (this.matchOperator("^")) {
      const right = this.parseAddition();
      if (left.kind !== "integer" || right.kind !== "integer") {
        throw new TypeError("位异或仅支持整数");
      }
      left = {
        value: Number(BigInt(left.value) ^ BigInt(right.value)),
        kind: "integer",
      };
    }
    return left;
  }

  private parseAddition(): NumericValue {
    let left = this.parseMultiplication();
    while (this.isOperator("+") || this.isOperator("-")) {
      const operator = this.advance().text;
      const right = this.parseMultiplication();
      left = {
        value: operator === "+" ? left.value + right.value : left.value - right.value,
        kind: left.kind === "integer" && right.kind === "integer" ? "integer" : "float",
      };
    }
    return left;
  }

  private parseMultiplication(): NumericValue {
    let left = this.parseUnary();
    while (this.isOperator("*") || this.isOperator("/")) {
      const operator = this.advance().text;
      const right = this.parseUnary();
      if (operator === "/" && right.value === 0) throw new Error("division by zero");
      left = {
        value: operator === "*" ? left.value * right.value : left.value / right.value,
        kind:
          operator === "*" && left.kind === "integer" && right.kind === "integer"
            ? "integer"
            : "float",
      };
    }
    return left;
  }

  private parseUnary(): NumericValue {
    if (this.matchOperator("-")) {
      const operand = this.parseUnary();
      return { value: -operand.value, kind: operand.kind };
    }
    // Python AST 会生成 UAdd，但原工具的运算符白名单没有 UAdd，因此保持为不支持。
    if (this.matchOperator("+")) throw new Error("不支持一元正号");
    return this.parsePower();
  }

  private parsePower(): NumericValue {
    const left = this.parsePrimary();
    if (!this.matchOperator("**")) return left;

    // 右侧重新进入 unary，使 2**-2 可用，同时自然实现 2**3**2 的右结合。
    const right = this.parseUnary();
    const value = left.value ** right.value;
    if (!Number.isFinite(value)) throw new RangeError("数值结果超出范围");
    return {
      value,
      kind:
        left.kind === "integer" && right.kind === "integer" && right.value >= 0
          ? "integer"
          : "float",
    };
  }

  private parsePrimary(): NumericValue {
    const token = this.advance();
    if (token.type === "number") {
      return {
        value: Number(token.text),
        kind: /[.eE]/.test(token.text) ? "float" : "integer",
      };
    }

    if (token.type === "leftParenthesis") {
      const value = this.parseXor();
      this.expect("rightParenthesis", "缺少右括号");
      return value;
    }

    if (token.type === "identifier") {
      if (this.current().type === "leftParenthesis") {
        this.advance();
        const args: NumericValue[] = [];
        if (this.current().type !== "rightParenthesis") {
          do {
            args.push(this.parseXor());
          } while (this.match("comma"));
        }
        this.expect("rightParenthesis", "函数调用缺少右括号");

        const mathFunction = this.functions[token.text];
        if (!mathFunction) throw new Error(`不支持的函数: ${token.text}`);
        return mathFunction(args);
      }

      const constant = this.constants[token.text];
      if (!constant) throw new Error(`未定义的变量: ${token.text}`);
      return { ...constant };
    }

    throw new SyntaxError(`不支持的表达式内容: ${token.text || token.type}`);
  }

  private current(): Token {
    return this.tokens[this.position];
  }

  private advance(): Token {
    return this.tokens[this.position++];
  }

  private isOperator(operator: string): boolean {
    const token = this.current();
    return token.type === "operator" && token.text === operator;
  }

  private matchOperator(operator: string): boolean {
    if (!this.isOperator(operator)) return false;
    this.advance();
    return true;
  }

  private match(type: Token["type"]): boolean {
    if (this.current().type !== type) return false;
    this.advance();
    return true;
  }

  private expect(type: Token["type"], message: string): Token {
    if (this.current().type !== type) throw new SyntaxError(message);
    return this.advance();
  }
}

function requireArity(name: string, args: NumericValue[], minimum: number, maximum = minimum): void {
  if (args.length < minimum || args.length > maximum) {
    const expected = minimum === maximum ? String(minimum) : `${minimum}-${maximum}`;
    throw new TypeError(`${name} 需要 ${expected} 个参数，实际收到 ${args.length} 个`);
  }
}

function floatFunction(
  name: string,
  operation: (first: number, second?: number) => number,
  minimum = 1,
  maximum = minimum,
): MathFunction {
  return (args) => {
    requireArity(name, args, minimum, maximum);
    const value = operation(args[0].value, args[1]?.value);
    if (!Number.isFinite(value)) throw new RangeError(`${name} 的结果超出范围`);
    return { value, kind: "float" };
  };
}

/** Python round 使用“银行家舍入”，与 JavaScript Math.round 的半数规则不同。 */
function pythonRound(value: number, digits = 0): number {
  const factor = 10 ** digits;
  const scaled = value * factor;
  const floor = Math.floor(scaled);
  const fraction = scaled - floor;
  const rounded =
    Math.abs(fraction - 0.5) < Number.EPSILON * Math.max(1, Math.abs(scaled))
      ? floor % 2 === 0
        ? floor
        : floor + 1
      : Math.round(scaled);
  return rounded / factor;
}

/** 内置计算器工具。 */
export class CalculatorTool extends Tool {
  private static readonly CONSTANTS: Readonly<Record<string, NumericValue>> = {
    pi: { value: Math.PI, kind: "float" },
    e: { value: Math.E, kind: "float" },
  };

  private static readonly FUNCTIONS: Readonly<Record<string, MathFunction>> = {
    abs: (args) => {
      requireArity("abs", args, 1);
      return { value: Math.abs(args[0].value), kind: args[0].kind };
    },
    round: (args) => {
      requireArity("round", args, 1, 2);
      const digits = args[1]?.value ?? 0;
      if (!Number.isInteger(digits)) throw new TypeError("round 的位数必须是整数");
      return {
        value: pythonRound(args[0].value, digits),
        kind: args.length === 1 ? "integer" : args[0].kind,
      };
    },
    max: (args) => {
      if (args.length === 0) throw new TypeError("max 至少需要一个参数");
      return args.reduce((best, current) => (current.value > best.value ? current : best));
    },
    min: (args) => {
      if (args.length === 0) throw new TypeError("min 至少需要一个参数");
      return args.reduce((best, current) => (current.value < best.value ? current : best));
    },
    // Python 原实现虽注册了 sum，但其 AST 求值器不支持列表；这里明确拒绝以避免错误理解。
    sum: () => {
      throw new TypeError("sum 需要列表参数，当前安全表达式语法不支持列表");
    },
    sqrt: floatFunction("sqrt", Math.sqrt),
    sin: floatFunction("sin", Math.sin),
    cos: floatFunction("cos", Math.cos),
    tan: floatFunction("tan", Math.tan),
    log: floatFunction(
      "log",
      (value, base = Math.E) => Math.log(value) / Math.log(base),
      1,
      2,
    ),
    exp: floatFunction("exp", Math.exp),
  };

  constructor() {
    super(
      "python_calculator",
      "执行数学计算。支持基本运算、数学函数等。例如：2+3*4, sqrt(16), sin(pi/2)等。",
    );
  }

  run(parameters: Record<string, unknown>): string {
    // 同时接受 input 和 expression，便于兼容旧调用方以及语义更明确的新调用方。
    const expression = String(parameters.input || parameters.expression || "");
    if (!expression) return "错误：计算表达式不能为空";

    console.log(`🧮 正在计算: ${expression}`);
    try {
      const tokens = new Lexer(expression).tokenize();
      const result = new ExpressionParser(
        tokens,
        CalculatorTool.CONSTANTS,
        CalculatorTool.FUNCTIONS,
      ).parse();
      const resultText = CalculatorTool.formatResult(result);
      console.log(`✅ 计算结果: ${resultText}`);
      return resultText;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const errorMessage = `计算失败: ${message}`;
      console.log(`❌ ${errorMessage}`);
      return errorMessage;
    }
  }

  getParameters(): ToolParameter[] {
    return [
      new ToolParameter({
        name: "input",
        type: "string",
        description: "要计算的数学表达式，支持基本运算和数学函数",
        required: true,
      }),
    ];
  }

  /** 尽量贴近 Python str(number)，包括整数、浮点整数以及非科学计数法结果。 */
  private static formatResult(result: NumericValue): string {
    if (result.kind === "float" && Number.isInteger(result.value)) return `${result.value}.0`;
    return String(result.value);
  }
}

/** 无需显式创建工具实例的便捷计算入口。 */
export function calculate(expression: string): string {
  return new CalculatorTool().run({ input: expression });
}
