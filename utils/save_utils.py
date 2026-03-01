import json
import os
import time
import re
from pathlib import Path

class MarkdownWriter:
    def __init__(
        self,
        problem: str,
        topic: str,
        log_dir: str = "logs",
        depth: int | None = None,
        node_id: int | None = None,
        *,
        file_prefix: str | None = None,
        markdown_file: str | None = None,
    ):
        self.problem = problem
        self.topic = topic
        self.depth = depth
        self.node_id = node_id
        self.file_prefix = file_prefix

        # 为每个 node 构造单独的日志目录：<task_dir>/node_{node_id}
        base_path = Path(log_dir)
        if self.node_id is not None:
            base_path = base_path / f"node_{self.node_id}"
        self.log_dir = str(base_path)

        os.makedirs(self.log_dir, exist_ok=True)

        # 如果指定了 markdown_file，就直接用该路径
        if markdown_file:
            self.markdown_file = str(markdown_file)
        else:
            self.markdown_file = self.get_markdown_file()

        self.buffer = []

        # 只有在文件不存在或为空时写入头部
        try:
            file_exists = os.path.exists(self.markdown_file)
            file_nonempty = file_exists and os.path.getsize(self.markdown_file) > 0
        except Exception:
            file_nonempty = False

        if not file_nonempty:
            header = [
                "# Topic\n",
                f"{self.topic}\n",
                "# Task\n",
                f"{self.problem}\n"
            ]
            self._write_lines(header)

    def get_markdown_file(self):
        timestamp = time.strftime('%m%d')
        if self.file_prefix:
            raw_prefix = str(self.file_prefix)
        else:
            raw_prefix = self.problem or 'problem'

        sanitized = re.sub(r'[^A-Za-z0-9\u4e00-\u9fff_-]+', '_', raw_prefix.strip())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        prefix = sanitized[:60] if sanitized else 'problem'

        extra_parts = []
        if self.node_id is not None:
            extra_parts.append(f"node{self.node_id}")
        extra = "_".join(extra_parts)

        if extra:
            filename = f"{timestamp}_{prefix}_{extra}.md"
        else:
            filename = f"{timestamp}_{prefix}.md"

        return os.path.join(self.log_dir, filename)

    def _write_lines(self, lines: list[str]):
        """写入文件并保存到内存 buffer"""
        with open(self.markdown_file, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
                self.buffer.append(line)

    def write_to_markdown(self, text: str, mode: str = 'supervisor'):
        """
        - Supervisor Response / Critic Response / Theoretician Response (##)
        - tool (expects text to include tool name/context) (#)
        """
        if text is None:
            text = ""

        mode = mode or ""
        if mode != "tool":
            text = text.replace("#", "\\#")  # 转义 Markdown #

        if mode in ('supervisor', 'Supervisor Response'):
            self._write_lines(["## Supervisor Response\n", f"{text}\n"])
        elif mode in ('supervisor_critic', 'critic', 'Critic Response'):
            self._write_lines(["## Critic Response\n", f"{text}\n"])
        elif mode in ('theoretician_response', 'theoretician', 'Theoretician Response'):
            self._write_lines(["## Theoretician Response\n", f"{text}\n"])
        elif mode == 'tool':
            self._write_lines([f"### Tool - {text}\n"])
        else:
            self._write_lines([f"## {mode}\n", f"{text}\n"])

    def log_tool_call(self, name: str, arguments: dict):
        payload = json.dumps(arguments or {}, ensure_ascii=False, indent=2)
        self._write_lines([
            f"### Tool - {name} call\n",
            "```json\n",
            f"{payload}\n",
            "```\n",
        ])

    def log_tool_result(self, name: str, result):
        body = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
        self._write_lines([
            f"### Tool - {name} result\n",
            "```\n",
            f"{body}\n",
            "```\n",
        ])

    def log_message(self, label: str, content: str):
        safe_label = label or "Message"
        self._write_lines([f"## {safe_label}\n", f"{content or ''}\n"])

    def get_buffer(self) -> str:
        """返回内存中完整 Markdown 内容"""
        return "".join(self.buffer)
