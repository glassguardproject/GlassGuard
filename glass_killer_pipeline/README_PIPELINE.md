# Glass Killer — 3-node pipeline (throughput build)

**Goal:** raise *throughput* by running the frame's stages as 3 separate processes
(3 GILs → true multicore). Latency = sum of stages (~1.2 s, one-frame-ish lag,
accepted); throughput = 1 / slowest stage.

This folder is a SEPARATE build. `../glass_killer_ros_node.py` (the working single
-process node) is untouched and remains the reference / fallback. Both import the
same `../batch_bigmask_4ray_randomopt.py` (bsp), so stage logic is reused, not
rewritten — each node calls the same bsp functions the monolith calls.

## Stage split (from the real `_pt()` markers in `_process`, post-0.05-voxel)

| stage / node        | monolith `_pt` markers covered                     | ~ms  |
|---------------------|----------------------------------------------------|------|
| **node1_detect**    | rgb, cloud_parse, pinhole_align, project, da2, sam3, splat | ~392 |
| **node2_geometry**  | doorway, rays, seed_extract, (small_overlay), assign, accum_seed, seed_pub, scene_depth, floor, clench | ~411 |
| **node3_mapping**   | publish_planes, global_map (merge + ground evict + spill evict + final tracker) | ~434 |

Balanced ⇒ throughput ≈ 1/0.434 ≈ **2.3 Hz** (vs 0.8 Hz single-process). `global_map`
(~320 ms) is the floor; a true 3 Hz needs node3 to also split spill‖ground internally
(safe: their per-plane state fields are disjoint — see notes below).

## Boundary data contracts (what crosses each hop)

Payloads are Python dicts of numpy arrays. FAT items (masks, cloud) blow past the
16 MB UDP DDS buffer, so we DON'T ship them over a topic — we drop each payload as a
pickle in `/dev/shm` (RAM) and publish only the *filename + header stamp* on a
`std_msgs/String` topic. Next node reads and unlinks it. See `pipeline_transport.py`.
Masks are `np.packbits`-ed (÷8) before pickling.

- **cloud+image → node1** : the provider's existing topics (same as the monolith:
  `/registered_scan`/last-scan, `/camera/image`, pose, terrain). node1 subscribes to
  these directly — no new contract here.
- **node1 → node2  (`/gkpipe/detect`)** : `{ stamp, pose7, big_masks(packbits),
  big_idx, mask_meta(colors/scores), pc_xyz(5cm), img_wh, pano_offset }`
- **node2 → node3  (`/gkpipe/geom`)** : `{ stamp, pose7, clench_by_mask, seed_records,
  mask_ray_records, floor_gate_xyz, floor_world, obst_world, glass_mask(packbits),
  pc_xyz, img_wh, pano_offset }`
- **node3 → world** : the existing output topics verbatim
  (`/glass_killer/global_planes`, `/glass_killer/final_global_planes`,
  `/added_obstacles`) — downstream planner/rviz unchanged.

Each payload carries its **capture-time pose** (pose7) so node3 evicts/places against
the pose the frame was actually taken at, not the robot's now-position. Pipeline lag
never corrupts geometry — it only delays delivery.

## Sync / backpressure
- Nodes key everything by `cloud_msg.header.stamp`; each stage stamps its output with
  the SAME stamp so a frame is traceable end-to-end.
- Drop-to-latest: if a node is busy when a new payload arrives, it keeps only the
  newest (mirrors the monolith's single-in-flight behavior). Stale `/dev/shm` pickles
  older than the newest are unlinked so RAM can't grow unbounded.

## Port status
- [x] transport (`pipeline_transport.py`) — /dev/shm payloads + packbits masks
- [x] node skeletons with real topics + bsp import + drop-to-latest plumbing
- [ ] node1: port `_process` lines ~1400–1540 (image/cloud → sam3 → splat)
- [ ] node2: port lines ~1543–1798 (rays…clench)
- [ ] node3: port lines ~1865–2000 (global_map) — reuse the disjoint-field
      spill‖ground split for the internal sub-parallel once single-thread parity holds
- [ ] launch + parity test vs the monolith on one bag (same GT eval numbers)

## Parity rule
Before trusting speed: the 3-node map on a fixed bag must match the single-node map
(same planes, same evictions) to eval tolerance. Only then tune throughput. The
monolith stays the scoring reference for the paper.
