import { describe, expect, test } from "bun:test";
import {
  applyEditToSeries,
  describeEdit,
  editFrameMask,
  featureGroupsFromRow,
  type NumericEdit,
} from "@/utils/numericEdit";

describe("featureGroupsFromRow", () => {
  test("groups keys by feature with dim indices in key order", () => {
    const groups = featureGroupsFromRow({
      timestamp: 0,
      "observation.state | odom_x": 1,
      "observation.state | odom_y": 2,
      "action | 0": 3,
      "action | 1": 4,
    });
    expect(groups).toHaveLength(2);
    const state = groups.find((g) => g.feature === "observation.state")!;
    expect(state.dims.map((d) => d.label)).toEqual(["odom_x", "odom_y"]);
    expect(state.dims.map((d) => d.index)).toEqual([0, 1]);
    const action = groups.find((g) => g.feature === "action")!;
    expect(action.dims.map((d) => d.index)).toEqual([0, 1]);
  });

  test("treats delimiter-less keys as scalar features", () => {
    const groups = featureGroupsFromRow({ timestamp: 0, reward: 1 });
    expect(groups).toEqual([
      {
        feature: "reward",
        dims: [{ key: "reward", label: "reward", index: 0 }],
      },
    ]);
  });

  test("never includes timestamp", () => {
    expect(featureGroupsFromRow({ timestamp: 1 })).toEqual([]);
  });
});

describe("editFrameMask", () => {
  const ts = Array.from({ length: 10 }, (_, i) => i * 0.1); // 10 fps, 1s

  test("null range selects every frame", () => {
    expect(editFrameMask(ts, 1, null).every(Boolean)).toBe(true);
  });

  test("fraction 0.5–1.0 selects the second half", () => {
    const mask = editFrameMask(ts, 1, { mode: "fraction", start: 0.5, end: 1 });
    expect(mask).toEqual([
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      true,
      true,
    ]);
  });

  test("seconds range selects by episode-relative timestamp", () => {
    const mask = editFrameMask(ts, 1, {
      mode: "seconds",
      start: 0.2,
      end: 0.4,
    });
    expect(mask.filter(Boolean).length).toBe(3); // frames at 0.2, 0.3, 0.4
    expect(mask[2] && mask[3] && mask[4]).toBe(true);
  });

  test("seconds range works with non-zero-based timestamps", () => {
    const shifted = ts.map((t) => t + 5);
    const mask = editFrameMask(shifted, 1, {
      mode: "seconds",
      start: 0.2,
      end: 0.4,
    });
    expect(mask[2] && mask[3] && mask[4]).toBe(true);
    expect(mask[0] || mask[9]).toBe(false);
  });
});

describe("applyEditToSeries", () => {
  const ts = Array.from({ length: 10 }, (_, i) => i * 0.1);
  const values = Array.from({ length: 10 }, (_, i) => i);

  test("set fixes the value in range, keeps the rest", () => {
    const out = applyEditToSeries(values, ts, 1, {
      op: "set",
      value: 1.0,
      range: { mode: "fraction", start: 0.5, end: 1 },
    });
    expect(out.slice(0, 5)).toEqual([0, 1, 2, 3, 4]);
    expect(out.slice(5)).toEqual([1, 1, 1, 1, 1]);
    expect(values[5]).toBe(5); // input not mutated
  });

  test("add offsets, scale multiplies", () => {
    const added = applyEditToSeries(values, ts, 1, {
      op: "add",
      value: 10,
      range: null,
    });
    expect(added[3]).toBe(13);
    const scaled = applyEditToSeries(values, ts, 1, {
      op: "scale",
      value: 2,
      range: null,
    });
    expect(scaled[3]).toBe(6);
  });
});

describe("describeEdit", () => {
  const base: NumericEdit = {
    id: "1",
    feature: "observation.state",
    dim: 2,
    dimLabel: "odom_x",
    episodeIndex: null,
    range: { mode: "fraction", start: 0.5, end: 1 },
    op: "set",
    value: 1,
  };

  test("describes a set-on-fraction edit", () => {
    expect(describeEdit(base)).toBe(
      "observation.state[odom_x] = 1 · 50%–100% · all eps",
    );
  });

  test("describes all-dims / single-episode / seconds", () => {
    expect(
      describeEdit({
        ...base,
        dim: null,
        dimLabel: null,
        episodeIndex: 3,
        range: { mode: "seconds", start: 1.5, end: 2 },
        op: "scale",
        value: 0.5,
      }),
    ).toBe("observation.state (all dims) ×= 0.5 · 1.50s–2.00s · ep 3");
  });
});
