export class MemoryConfig {
    // public readonly maxCapacity: number;
    // public readonly importanceThreshold: number;
    // public readonly decayFactor: number;

    // public readonly workingMemoryCapacity: number
    // public readonly  workingMemoryTTL: Number

    constructor(
        maxCapacity: number = 1000,
        importanceThreshold: number = 0.7,
        decayFactor: number = 0.05,
        workingMemoryCapacity: number = 50,
        workingMemoryTTL: number = 60,
        workingMemoryTokens: number = 1000,
    ) {
        this.maxCapacity = maxCapacity;
        this.importanceThreshold = importanceThreshold;
        this.decayFactor = decayFactor;
        this.workingMemoryCapacity = workingMemoryCapacity;
        this.workingMemoryTTL = workingMemoryTTL;
        this.workingMemoryTokens = workingMemoryTokens;
    }

    public readonly maxCapacity: number;
    public readonly importanceThreshold: number;
    public readonly decayFactor: number;
    public readonly workingMemoryCapacity: number;
    public readonly workingMemoryTTL: number;
    public readonly workingMemoryTokens: number;
}
