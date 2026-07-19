export class MemoryConfig {
    // public readonly maxCapacity: number;
    // public readonly importanceThreshold: number;
    // public readonly decayFactor: number;

    // public readonly workingMemoryCapacity: number
    // public readonly  workingMemoryTTL: Number

    constructor(
        public readonly maxCapacity: number = 1000,
        public readonly importanceThreshold: number = 0.7,
        public readonly decayFactor: number = 0.05,
        public readonly workingMemoryCapacity: number = 50,
        public readonly workingMemoryTTL: number = 60,
        public readonly workingMemoryTokens: number = 1000,
    ) {
    }
}
