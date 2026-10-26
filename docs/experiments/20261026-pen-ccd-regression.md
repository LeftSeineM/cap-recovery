# 笔 000002：CPU CCD 场景回归

关联：PR #74，问题背景见 #43。

## 2026-10-26 结论

CPU 薄板 12/12 通过，组合后的 USD 有效属性中只有五个刚体的 `enableCCD` 从 false 变为 true。当前 main（含 #69）上，000000 由 19/20 提高到 20/20，000003 保持 19/20，000002 由 20/20 降为 19/20，未通过每个场景成功数不低于旧资产的要求。没有调整物理参数或用重跑替换主批次失败。

## 结果

两组运行代码均为 `015b2da91ba759301d921a7e0e01c0ed937d714a`（#74），包含 main `f5c6be74bcc4ff0685fb7282c388b677a8be6ae5`（#69）。旧组加载共享笔，CCD 组加载隔离根；全部 120 条 episode 有结果，无进程错误，seed 均为 0:20。

| 场景 | 旧资产 | CCD | 失败 seed |
|---|---:|---:|---|
| 000002 | 20/20 | 19/20 | CCD 18；低于旧资产，未通过不回退要求 |
| 000003 | 19/20 | 19/20 | 两组均 13，失败机制不同 |
| 000000 | 19/20 | 20/20 | 旧 18 |

性能按每条 episode 的 `wall_profile.physics_step` 除以 calls，再对双方共同成功的同 seed 等权平均，每场景 19 对。每控制步含 10 个 4 ms 物理步。进程 CPU 是所有进程线程的累计 CPU 时间；配对 bootstrap 为 20000 次，固定随机种子 20261026。

| 场景 | wall 旧→CCD（ms/控制步） | 变化及 95% CI | 进程 CPU 旧→CCD（ms/控制步） | 变化及 95% CI |
|---|---:|---:|---:|---:|
| 000002 | 52.458→52.814 | +0.68% [−5.02%, +6.12%] | 61.641→62.252 | +0.99% [−4.53%, +6.14%] |
| 000003 | 53.766→51.913 | −3.45% [−7.49%, +1.44%] | 63.133→60.960 | −3.44% [−6.61%, +0.18%] |
| 000000 | 55.605→52.597 | −5.41% [−8.75%, −1.82%] | 66.195→62.503 | −5.58% [−8.56%, −2.32%] |

| 场景/资产 | load/core p10 / p50 / p90 / max | episode 进程 CPU/wall p50 |
|---|---|---:|
| 000002/旧 | 0.464 / 0.565 / 0.676 / 0.997 | 1.777 |
| 000002/CCD | 0.411 / 0.558 / 0.696 / 1.046 | 1.818 |
| 000003/旧 | 0.473 / 0.561 / 0.697 / 1.063 | 2.232 |
| 000003/CCD | 0.473 / 0.563 / 0.717 / 1.063 | 2.260 |
| 000000/旧 | 0.475 / 0.595 / 0.674 / 0.742 | 1.315 |
| 000000/CCD | 0.439 / 0.478 / 0.636 / 0.710 | 1.337 |

000000 的两组负载不同，下降不能解释为 CCD 提速；000002、000003 的变化区间均跨零。`physics_step` 包含动作转换、物理推进与记录，不把整条 episode 的规划成本归为 CCD 成本。

## 失败机制

000002 CCD seed 18 第二次从桌面抓笔，左手接触面积为 9.75 mm²（旧组 103.25 mm²）。step 1922 提起结束时，笔相对 TCP 漂移 0.056 mm；step 1945 已漂移 58.68 mm，夹爪直到 step 2004 才开。笔在闭合夹爪中滑落，最终停在桌面，属于 CCD 组独有失败。前序接触演化、桌面笔姿态和抓取候选已经分叉，不能声称两组执行了相同抓取路径。

从该失败的 frame 36（place 入口）取同一快照，新旧资产各做 3 次重放：旧 2/3、CCD 3/3，原随机流均成功，旧组 repeat 1 出现同类失败。诊断没有稳定复现原失败，也不证明 CCD 会直接破坏同一弱持握；快照不包含完整 PhysX 接触求解缓存。这些结果不替换主批次的 20/20→19/20。

000003 两组均 seed 13 失败，但旧组是放置 IK 不可达（114 个候选、228 次 IK、planned=0）；CCD 组搬运时推动柜体，柜体 yaw 从 21.965° 变为 59.119°，平移约 125 mm，放置却仍使用移柜前的世界区域。开爪前笔在实际抽屉坐标 x=−237.77 mm，而底板半宽 129 mm，笔落在底板范围外的桌面，没有穿过底板。该条直到开爪前相对 TCP 漂移仅 0.861 mm，不是搬运滑落；它暴露了环境被推动后放置区域未重新求值的问题。

000000 旧 seed 18 抽屉最终开口 5.803016 mm，CCD 同 seed 成功。

## 环境与输入

AMD EPYC 7542，128 逻辑核，RTX 4090；Isaac Sim 5.1.0、Isaac Lab 2.3.2.post1、CPU PhysX。六批同一主机同一窗口提交（2026-10-26 15:58:40 UTC+8），每批 4 个分片，与其他任务共享资源，由显存队列调度。物理 dt=0.004 s，每控制步 10 个物理步。

000000、000002 使用共享初态，000003 使用 #60 的同批 20 seed 初态；其 `--input` 根包含 `000003/initial_states.hdf5`。三个初态 HDF5 的 SHA256：

- 000000：`4f938628e060e902150b804c2b92eaaa0b93d0b6c25ab7508912b822ed2f656e`。
- 000002：`b4d986964fc30aab0c6b8744f22c184977ff448a9bcfb93c5debe307925466be`。
- 000003：`40989a9f53c18eea26f19cd5ea61637a22b3c2d259e750b4db0c0b66dff5b36e`。

隔离根只构建 pen/000002，其余资产软链接引用共享根；有效 USD 审计 3429 项，只有 body、cap、cap_ball1..3 五处 CCD 布尔差异。共享笔 1632 个文件和三份初态在前后校验中均未改变。薄板为 6/9/12 cm × 0/20/40/60° 共 12 组合，最低钢珠球心 +0.7999997 mm，沿用球心不低于 −0.1 mm 的夹具判据。粗筛单元测试 9/9 通过；hardware instructions 因 EACCES 未获取。

## 复现

`ISOLATED` 为新的隔离资产根，其他资产软链接指向共享根。`SCENE` 依次取 000002、000003、000000；`VARIANT` 取 old 或 ccd，`ASSETS` 分别指共享根或隔离根，每对同时启动。000003 命令追加 `--input ${INITIALS_000003}`，该初态 SHA256 须与上述一致。

```bash
git checkout 015b2da91ba759301d921a7e0e01c0ed937d714a
blender -b --python-exit-code 1 -P scripts/assets/build_assets.py -- --output ${ISOLATED}/objects --assets pen/000002
uv run python scripts/assets/build_physics.py --output ${ISOLATED}/objects --assets pen/000002 --gpu 0
uv run python tests/physics/isaac_pen_thin_shelf.py --asset ${ISOLATED}/objects/pen/000002/object.usdz --output outputs/pen-ccd-thin-repeat --gpu 0
uv run python scripts/scenes/run_expert.py ${SCENE} --physics isaac-cpu --dev --profile --seeds 0:20 --jobs 4 --gpus 0 1 2 3 --name pen-ccd-current-${VARIANT} --assets ${ASSETS}
uv run python scripts/experts/failure_snapshots.py ${BATCH_000002_CCD} --seeds 18 --frame 36 --name pen-ccd-seed18-place
# SNAPSHOT 为上一命令的输出目录，诊断重放也比较共享根与隔离根。
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --profile --steps 5 --from ${SNAPSHOT} --seeds 18 --repeats 3 --jobs 1 --gpus 0 --name pen-ccd-seed18-replay-${VARIANT} --assets ${ASSETS}
```

合并后由维护者决定是否重建共享资产，本轮未覆盖：

```bash
uv run python scripts/assets/build_physics.py --source asset_sources/objects --output assets/objects --assets pen/000002 --gpu 0
```
