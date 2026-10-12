#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

def main():
    p1 = Path("outputs/semantic_mapping_my_desk_recording_01/semantic_map.json")
    p2 = Path("outputs/semantic_mapping_my_desk_recording_02/semantic_map.json")

    with open(p1, "r", encoding="utf-8") as f:
        map1 = json.load(f)
    with open(p2, "r", encoding="utf-8") as f:
        map2 = json.load(f)

    ents1 = {k: v for k, v in map1.get("entities", {}).items() if v.get("is_confirmed")}
    ents2 = {k: v for k, v in map2.get("entities", {}).items() if v.get("is_confirmed")}

    print(f"Run 1 confirmed: {len(ents1)}, Run 2 confirmed: {len(ents2)}")

    # Test identity alignment (R = I, t = 0)
    matches_identity = []
    for k1, v1 in ents1.items():
        c1 = np.array(v1["centroid_floor"])
        cls1 = v1["class_name"]
        for k2, v2 in ents2.items():
            if v2["class_name"] != cls1:
                continue
            c2 = np.array(v2["centroid_floor"])
            dist = np.linalg.norm(c1[:2] - c2[:2])
            if dist < 0.6:  # within 60 cm
                matches_identity.append((k1, k2, cls1, dist, c1[:2].tolist(), c2[:2].tolist()))

    print(f"\nDirect Identity Matches (dist < 0.6m): {len(matches_identity)}")
    for m in sorted(matches_identity, key=lambda x: x[3])[:15]:
        print(f"  {m[0]} <-> {m[1]} ({m[2]}): dist={m[3]:.3f}m | c1={m[4]} c2={m[5]}")

    # Test small search for rotation theta and translation t (grid search within +/- 15 deg, +/- 1m)
    best_count = len(matches_identity)
    best_theta = 0.0
    best_t = np.zeros(2)

    for deg in np.linspace(-15, 15, 31):
        rad = np.radians(deg)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        for tx in np.linspace(-0.8, 0.8, 17):
            for ty in np.linspace(-0.8, 0.8, 17):
                t = np.array([tx, ty])
                cnt = 0
                for k1, v1 in ents1.items():
                    c1 = np.array(v1["centroid_floor"][:2])
                    cls1 = v1["class_name"]
                    for k2, v2 in ents2.items():
                        if v2["class_name"] != cls1:
                            continue
                        c2 = R @ np.array(v2["centroid_floor"][:2]) + t
                        if np.linalg.norm(c1 - c2) < 0.5:
                            cnt += 1
                            break
                if cnt > best_count:
                    best_count = cnt
                    best_theta = deg
                    best_t = t

    print(f"\nOptimal Grid Search Alignment:")
    print(f"  Best Matches: {best_count}")
    print(f"  Rotation:     {best_theta:.1f} deg")
    print(f"  Translation:  [{best_t[0]:.2f}, {best_t[1]:.2f}] m")

if __name__ == "__main__":
    main()
