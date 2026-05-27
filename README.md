# 本地视频链接转文稿

这个小工具不使用云端语音识别 API。流程是：

1. `yt-dlp` 从视频链接下载音频。
2. `ffmpeg` 把音频转成 16kHz 单声道 mp3。
3. `faster-whisper` 在本机运行 Whisper 模型，输出 `txt`、`srt`、`json`。

## Conda 安装方式

推荐使用 Python 3.10 的 conda 环境。`faster-whisper` 依赖的底层库在 Python 3.10/3.11 上通常更稳。

```powershell
conda create -n v2s python=3.10 -y
conda activate v2s
```

安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 里包含 `imageio-ffmpeg`，会在当前 Python/conda 环境中提供可用的 ffmpeg。通常不需要再单独运行 `conda install ffmpeg` 或 `winget install ffmpeg`。

检查依赖是否安装成功：

```powershell
python --version
python -m pip show faster-whisper
python -m pip show imageio-ffmpeg
python -m pip show yt-dlp
```

## venv 安装方式

如果不用 conda，也可以用 Python 自带虚拟环境。先安装 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 使用

默认不写 `--language` 时，程序会自动识别视频里的语音语言。

```powershell
python transcribe_video.py "https://example.com/video-url"
```

中文视频：

```powershell
python transcribe_video.py "视频链接" --language zh --model small
```

英文视频：

```powershell
python transcribe_video.py "视频链接" --language en --model small
```

B 站视频也可以尝试直接输入链接，只要 `yt-dlp` 能下载该视频即可：

```powershell
python transcribe_video.py "https://www.bilibili.com/video/BVxxxxxx" --language zh --model small
```

如果是中英混合内容，可以不写 `--language`，让模型自动判断：

```powershell
python transcribe_video.py "视频链接" --model small
```

## 语言参数

`--language` 用来指定视频里的主要语音语言。这个参数不是必填项；不写就是自动识别语言。指定后通常更稳定，尤其适合整段都是中文、英文、日文这类单一语言的视频。

自动识别：

```powershell
python transcribe_video.py "视频链接" --model small
```

常用语言：

```powershell
python transcribe_video.py "视频链接" --language zh --model small
python transcribe_video.py "视频链接" --language en --model small
python transcribe_video.py "视频链接" --language ja --model small
python transcribe_video.py "视频链接" --language ko --model small
python transcribe_video.py "视频链接" --language fr --model small
python transcribe_video.py "视频链接" --language de --model small
python transcribe_video.py "视频链接" --language es --model small
```

常见语言代码：

- `zh`：中文
- `en`：英文
- `ja`：日文
- `ko`：韩文
- `fr`：法文
- `de`：德文
- `es`：西班牙文
- `ru`：俄文
- `it`：意大利文
- `pt`：葡萄牙文

## GPU 使用

如果你有 NVIDIA 显卡，并且已经配置好 CUDA 相关环境，可以尝试：

```powershell
python transcribe_video.py "视频链接" --device cuda --compute-type float16 --model medium
```

CPU 用户建议保持默认：

```powershell
python transcribe_video.py "视频链接" --device cpu --compute-type int8 --model small
```

## 模型选择

- `tiny` / `base`：最快，准确率较低，适合先确认流程能跑通。
- `small`：默认推荐，速度和准确率比较均衡。
- `medium`：更准，但明显更慢、更吃内存。
- `large-v3`：质量更高，适合 GPU。

第一次运行会从 Hugging Face 下载模型文件。下载完成后，同一个模型会缓存在本地，之后可以离线识别。

## 输出

输出文件在 `outputs/`：

- `.txt`：带时间戳的文稿，方便阅读。
- `.srt`：字幕文件，可以导入播放器。
- `.json`：结构化片段，方便后续做摘要、搜索或剪辑。

示例：

```text
视频标题

[00:00:01.200 -> 00:00:04.800] This is the first sentence.
[00:00:04.900 -> 00:00:08.300] This is the next sentence.
```

## 常见问题

### No module named 'faster_whisper'

说明当前 Python 环境没有安装依赖。先确认已经激活 conda 环境：

```powershell
conda activate v2s
python -m pip install -r requirements.txt
```

### 找不到 ffmpeg

本项目会按下面顺序自动寻找 ffmpeg：

1. 系统 PATH 里的 `ffmpeg`。
2. 当前 conda/venv 环境里的 `ffmpeg`。
3. `imageio-ffmpeg` 随 Python 依赖提供的 ffmpeg。

如果仍然提示找不到 ffmpeg，先确认 Python 依赖已经安装到当前环境：

```powershell
python -m pip install -r requirements.txt
python -m pip show imageio-ffmpeg
```

也可以让程序使用指定路径的 ffmpeg：

```powershell
python transcribe_video.py "视频链接" --ffmpeg "C:\path\to\ffmpeg.exe"
```

如果你仍想把 ffmpeg 作为 conda 包安装，可以手动尝试：

```powershell
conda install -c conda-forge ffmpeg -y
```

但 Windows 上 conda-forge 安装 `ffmpeg` 有时会遇到 `gdk-pixbuf`、`Rolling back transaction`、`UnicodeDecodeError('gbk', ...)` 之类的依赖脚本问题。现在脚本已经不依赖这条安装路径，优先使用 `imageio-ffmpeg` 即可。

### Hugging Face 提示 HF_TOKEN

这不是错误。模型第一次运行时需要从 Hugging Face 下载，匿名下载公开模型通常也可以，只是速度和限流可能不如登录账号。下载完成后，识别是在本地运行，不会把音频传到云端 API。

### Windows symlink warning

这是 Hugging Face 缓存模型时的 Windows 提示，不影响运行。它只是说明缓存可能多占一点磁盘空间。

## 说明

这不是 OpenAI API 方案，所以不需要 API Key。它更像播放器本地字幕识别功能：模型在你的电脑上运行，速度取决于 CPU/GPU 和选择的模型大小。
