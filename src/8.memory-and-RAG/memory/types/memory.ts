import { MemoryConfig } from "../config";
import { MemoryStore } from "../storage/store";


export abstract class Memory {
    constructor(config: MemoryConfig, store: MemoryStore) { }
}