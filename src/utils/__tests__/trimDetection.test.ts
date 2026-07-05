import { describe, expect, test } from "bun:test";
import {
  computeMotionProfile,
  detectActiveRange,
  getMotionKeys,
  isMeaningfulTrim,
} from "@/utils/trimDetection";

// 10 fps episode: static head [0, headFrames), sine motion, static tail.
// Noise simulates sensor jitter in the "static" parts.
function makeRows({
  frames = 100,
  fps = 10,
  headFrames = 20,
  tailFrames = 20,
  noise = 0.001,
  keys = ["action | 0", "action | 1"],
}: {
  frames?: number;
  fps?: number;
  headFrames?: number;
  tailFrames?: number;
  noise?: number;
  keys?: string[];
} = {}): Record<string, number>[] {
  const rows: Record<string, number>[] = [];
  for (let i = 0; i < frames; i++) {
    const active = i >= headFrames && i < frames - tailFrames;
    const row: Record<string, number> = { timestamp: i / fps };
    keys.forEach((key, k) => {
      // Deterministic "noise" so the test is reproducible.
      const jitter = noise * Math.sin(i * 7.13 + k * 3.7);
      row[key] = active ? Math.sin((i - headFrames) * 0.3 + k) : jitter;
    });
    rows.push(row);
  }
  return rows;
}

describe("getMotionKeys", () => {
  test("prefers action keys over state keys", () => {
    const keys = getMotionKeys({
      timestamp: 0,
      "action | 0": 1,
      "observation.state | 0": 2,
    });
    expect(keys).toEqual(["action | 0"]);
  });

  test("falls back to state keys when no actions", () => {
    const keys = getMotionKeys({
      timestamp: 0,
      "observation.state | 0": 2,
      "observation.state | 1": 3,
    });
    expect(keys).toEqual(["observation.state | 0", "observation.state | 1"]);
  });

  test("never returns timestamp", () => {
    expect(getMotionKeys({ timestamp: 0 })).toEqual([]);
  });
});

describe("computeMotionProfile", () => {
  test("returns null for too few rows", () => {
    expect(computeMotionProfile([{ timestamp: 0 }], 1)).toBeNull();
  });

  test("motion is high in the active region and low in static parts", () => {
    const rows = makeRows();
    const profile = computeMotionProfile(rows, 9.9);
    expect(profile).not.toBeNull();
    const { motion } = profile!;
    const headMax = Math.max(...motion.slice(0, 15));
    const activeMax = Math.max(...motion.slice(25, 75));
    expect(activeMax).toBeGreaterThan(headMax * 10);
  });

  test("uses index-based timestamps when rows lack a timestamp key", () => {
    const rows = makeRows().map((row) => {
      const rest = { ...row };
      delete rest.timestamp;
      return rest;
    });
    const profile = computeMotionProfile(rows, 9.9);
    expect(profile).not.toBeNull();
    expect(profile!.timestamps[0]).toBe(0);
    expect(profile!.timestamps[99]).toBeCloseTo(9.9, 5);
  });
});

describe("detectActiveRange", () => {
  test("detects static head and tail", () => {
    const rows = makeRows(); // active frames 20..79 → 2.0s .. 7.9s
    const range = detectActiveRange(rows, 9.9, { paddingSeconds: 0 });
    expect(range).not.toBeNull();
    // Smoothing spreads the onset by up to half the window (2 frames = 0.2s).
    expect(range!.start).toBeGreaterThanOrEqual(1.7);
    expect(range!.start).toBeLessThanOrEqual(2.1);
    expect(range!.end).toBeGreaterThanOrEqual(7.8);
    expect(range!.end).toBeLessThanOrEqual(8.3);
  });

  test("applies padding, clamped to the episode bounds", () => {
    const rows = makeRows();
    const padded = detectActiveRange(rows, 9.9, { paddingSeconds: 0.5 })!;
    const bare = detectActiveRange(rows, 9.9, { paddingSeconds: 0 })!;
    expect(padded.start).toBeCloseTo(Math.max(0, bare.start - 0.5), 5);
    expect(padded.end).toBeCloseTo(Math.min(9.9, bare.end + 0.5), 5);

    const huge = detectActiveRange(rows, 9.9, { paddingSeconds: 100 })!;
    expect(huge.start).toBe(0);
    expect(huge.end).toBe(9.9);
  });

  test("returns null for an all-static episode", () => {
    const rows = makeRows({ headFrames: 100, tailFrames: 0, noise: 0 });
    expect(detectActiveRange(rows, 9.9)).toBeNull();
  });

  test("covers the whole episode when motion never stops", () => {
    const rows = makeRows({ headFrames: 0, tailFrames: 0 });
    const range = detectActiveRange(rows, 9.9, { paddingSeconds: 0.25 })!;
    expect(range.start).toBeLessThanOrEqual(0.5);
    expect(range.end).toBeGreaterThanOrEqual(9.4);
  });

  test("higher sensitivity trims tighter than lower sensitivity", () => {
    const rows = makeRows({ noise: 0.02 });
    const loose = detectActiveRange(rows, 9.9, {
      sensitivity: 0.05,
      paddingSeconds: 0,
    })!;
    const tight = detectActiveRange(rows, 9.9, {
      sensitivity: 0.4,
      paddingSeconds: 0,
    })!;
    expect(tight.start).toBeGreaterThanOrEqual(loose.start);
    expect(tight.end).toBeLessThanOrEqual(loose.end);
  });
});

describe("isMeaningfulTrim", () => {
  test("true when the head is cut", () => {
    expect(isMeaningfulTrim({ start: 1, end: 10 }, 10)).toBe(true);
  });
  test("true when the tail is cut", () => {
    expect(isMeaningfulTrim({ start: 0, end: 8 }, 10)).toBe(true);
  });
  test("false when nothing meaningful is cut", () => {
    expect(isMeaningfulTrim({ start: 0.01, end: 9.95 }, 10)).toBe(false);
  });
});
