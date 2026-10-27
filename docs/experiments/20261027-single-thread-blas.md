# 专家 episode 单线程 BLAS

Isaac 启动后，专家进程里的 NumPy、SciPy 按 32 线程运行 BLAS，启动命令里的 `OPENBLAS_NUM_THREADS=1` 不生效。多个 episode 并发时，这些线程空转、互相争抢 CPU。把专家 episode 限制为单线程 BLAS 后（#85），三个场景的进程 CPU time 降低 48%–65%，wall time 不增加，成功数持平。

关联：#43，PR #85。

## 关键问题与结论

- 问题：专家 episode 的进程 CPU time 远高于调用线程的 CPU time，而且随负载起伏。多出来的部分来自哪里，能否去掉。
- 结论：
  - 根因：`SimulationApp._start_app()` 把 `OPENBLAS_NUM_THREADS` 改写为 CPU 线程上限（默认 32），之后加载的 NumPy、SciPy 的 OpenBLAS 都按 32 线程初始化。OpenMP 不受影响，仍为 1。
  - 修复：`Generator.run()` 和 `Evaluator.trial()` 入口用 `threadpoolctl` 把 BLAS 限制为 1 线程，报告的 `native_math_threadpools` 记录生效的线程池。
  - 效果：进程 CPU time 降低 48%、55%、65%（000002、000003、000103），非 PhysX 线程 CPU 基本不变，说明省下的是 BLAS 线程空转，不是计算本身。
  - 000003 先跑的 20 seed 中修复组 wall time 高 29%，来自 seed 8、9 抓笔没抓牢后的重试。改跑 100 seed 后两边 wall time 持平，含重试的 episode main 13 条、修复 11 条，原来的长尾是偶发。

## 结果

main `511d9eb`（#81）与修复 `734cba9`（#85）同时运行，均值含失败条目：

| 场景 | seed | 成功 main / 修复 | 进程 CPU time (s) main → 修复 | 非 PhysX 线程 CPU (s) main → 修复 | wall time (s) main → 修复 |
| --- | --- | --- | --- | --- | --- |
| 000002 | 0:20 | 20 / 20 | 435.1 → 226.5（−47.9%） | 107.2 → 100.1 | 246.8 → 218.8（−11.3%） |
| 000003 | 0:100 | 96 / 95 | 513.4 → 232.6（−54.7%） | 149.7 → 148.6 | 215.2 → 214.9（−0.1%） |
| 000103 | 0:20 | 3 / 3 | 3214.4 → 1131.4（−64.8%） | 980.7 → 924.3 | 1183.9 → 1077.4（−9.0%） |

000003 先跑的 seed 0:20：成功 19 / 19，进程 CPU time 427.3 → 259.8 s，非 PhysX 线程 CPU 117.5 → 163.5 s，wall time 190.7 → 246.6 s。100 seed 中修复多失败的一条是 seed 59（笔没放进抽屉）。

非 PhysX 线程 CPU 为 `profile_totals.thread_cpu_s - profile_totals.process_spans_thread_cpu_s`，不含 BLAS 工作线程。

## 复现

    git checkout <commit>
    uv run python scripts/scenes/run_expert.py <场景> --physics isaac-cpu --dev --profile --seeds <seed> [--input <初态>]

000002 用场景自带的初态。000003、000103 的初态分别来自 `runs/expert/000003/20261025-task4-scene003-initials100`（前 20 个与 seed 0:20 那次所用的相同）和 `assets/experiments/issue43-103-layout3-20261025`，不能由这些 commit 重建，这两个场景的结果是历史观察。

环境：双路 AMD EPYC 7542（容器可见 56–128 个逻辑 CPU），RTX 4090，与其他实验共享机器；每批 4 个并发进程。
