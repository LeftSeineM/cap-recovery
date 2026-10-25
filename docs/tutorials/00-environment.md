# 00 环境准备

所有命令都在仓库根目录执行。

## 系统工具

- [uv](https://docs.astral.sh/uv/)
- FFmpeg：`sudo apt install ffmpeg`
- Blender 4.5 LTS（实测 4.5.14）：从 [Blender 官网](https://www.blender.org/download/lts/) 下载并解压，把 `blender` 加入 `PATH`

## Python 环境

```bash
uv sync --locked --all-extras
```

其中包含 Isaac Sim 5.1 和 Isaac Lab，约 19 GB；之后一律用 `uv run python` 运行，不需要另装仿真器。只做第 1 章建模时，`uv sync --locked` 就够了。各依赖组见 [README](../../README.md#开发环境)。

## Isaac Sim 许可

首次运行前接受 EULA，可以写进 shell 配置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## 检查

```bash
blender --version
ffmpeg -version
uv run python -c "import table_1000"
```

开发机器人时还需要登录 Hugging Face 来下载厂商原始包，见[第 3 章](03-robot-import.md#命令)。
