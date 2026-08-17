from __future__ import annotations

from pipeline.artifacts import (
    prompt_artifacts_from_outputs,
    render_prompt_artifact,
    write_prompts,
)


def _output(item_id: str = "batch/1") -> dict:
    return {
        "item_id": item_id,
        "source_files": ["folder/etl.sas"],
        "target_language": "PySpark",
        "prompt": "translate etl",
        "prompt_messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Translate SAS exactly.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "system", "content": "Use INTNX semantics."},
            {"role": "human", "content": "translate etl"},
        ],
    }


def test_effective_prompt_markdown_preserves_roles_and_structured_content():
    artifact = render_prompt_artifact(_output())

    assert "# Effective prompt: batch/1" in artifact
    assert "## 1. System" in artifact
    assert "## 3. Human" in artifact
    assert '"cache_control"' in artifact
    assert "Translate SAS exactly." in artifact
    assert "translate etl" in artifact


def test_prompt_artifact_names_are_safe_and_collision_free():
    artifacts = prompt_artifacts_from_outputs(
        [_output("folder/item"), _output("folder:item")]
    )

    assert list(artifacts) == ["folder_item.md", "folder_item_2.md"]


def test_write_prompts_uses_its_own_output_subdirectory(tmp_path):
    (path,) = write_prompts([_output()], tmp_path)

    assert path == tmp_path / "prompts" / "batch_1.md"
    assert path.read_text(encoding="utf-8").startswith("# Effective prompt")


def test_resumed_item_is_labelled_and_does_not_overwrite_exact_artifact(tmp_path):
    output = _output()
    (path,) = write_prompts([output], tmp_path)
    exact = path.read_text(encoding="utf-8")
    output["prompt_messages"] = None

    (resumed_path,) = write_prompts([output], tmp_path)

    assert resumed_path == path
    assert resumed_path.read_text(encoding="utf-8") == exact


def test_resumed_item_without_prior_artifact_is_not_called_effective(tmp_path):
    output = _output()
    output["prompt_messages"] = None

    (path,) = write_prompts([output], tmp_path)
    artifact = path.read_text(encoding="utf-8")

    assert artifact.startswith("# Prompt unavailable")
    assert "re-derived item request" in artifact
    assert "# Effective prompt" not in artifact
