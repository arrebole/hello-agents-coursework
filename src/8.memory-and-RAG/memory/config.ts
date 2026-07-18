export class MemoryConfig {
    maxCapacity: number;
    importanceThreshold: number;
    decayFactor: number;

    constructor(
        maxCapacity: number = 1000,
        importanceThreshold: number = 0.7,
        decayFactor: number = 0.05,
    ) {
        this.maxCapacity = maxCapacity;
        this.importanceThreshold = importanceThreshold;
        this.decayFactor = decayFactor;
    }
}
