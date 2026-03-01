from pathlib import Path
from typing import List

import yaml

current_dir = Path(__file__).resolve().parent
TECHNIQUES_ROOT = current_dir.parent / "LANDAU" / "techniques"


class _PrettyDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_PrettyDumper.add_representer(str, _str_presenter)


def remove_yaml_comments_and_prettify(yaml_text: str) -> str:
    data = yaml.safe_load(yaml_text)
    return yaml.dump(
        data,
        Dumper=_PrettyDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()


def _resolve_technique_path(technique_id: str) -> Path | None:
    categories = ["general", "mathematics"]
    filenames = ["technique.yaml"]
    for category in categories:
        for filename in filenames:
            p = TECHNIQUES_ROOT / category / technique_id / filename
            if p.exists():
                return p
    return None


def load_technique_specs(technique_ids: List[str]) -> str:
    seen = set()
    uniq: List[str] = []
    for tid in technique_ids:
        if tid not in seen:
            seen.add(tid)
            uniq.append(tid)

    blocks = [
        "[TECHNIQUE FULL]",
        "The following content is Technique specifications (not ordinary tool output). "
        "Treat them as authoritative procedural knowledge and follow them when solving the task.",
        "",
    ]

    for technique_id in uniq:
        p = _resolve_technique_path(technique_id)
        if p is None:
            raise FileNotFoundError(f"Technique not found: {technique_id}")

        yaml_text = p.read_text(encoding="utf-8")
        yaml_text = remove_yaml_comments_and_prettify(yaml_text)

        blocks.append(
            "<TECHNIQUE_FULL>\n"
            f"{yaml_text}\n"
            "</TECHNIQUE_FULL>"
        )
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"
