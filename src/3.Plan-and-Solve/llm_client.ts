import OpenAI from 'openai';
import { ChatCompletionMessageParam } from 'openai/resources';

// 大模型调用
export class LLMClient {
  private model: string
  private client: OpenAI

  constructor(model: string, api_key: string, base_url: string) {
    this.model = model;
    this.client = new OpenAI({
      baseURL: base_url,
      apiKey: api_key,
    });
  }

  public async generate(prompt: string, system_prompt: string): Promise<string> {
    try {
      const messages: ChatCompletionMessageParam[] = [
        { 'role': 'system', 'content': system_prompt },
        { 'role': 'user', 'content': prompt }
      ];

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: messages,
        stream: false,
      })
      return response.choices[0].message.content!
    } catch (e) {
      return `错误:调用语言模型服务时出错。${e}`
    }
  }

  public async think(messages: ChatCompletionMessageParam[]) {
    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: messages,
        stream: false,
      });
      return response.choices[0].message.content!
    } catch (e) {
      return `错误:调用语言模型服务时出错。${e}`
    }
  }
}