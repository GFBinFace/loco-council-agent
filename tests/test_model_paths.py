"""测试本地模型路径解析与就绪检查。

重点覆盖两类风险：

1. 缓存路径解析出空字符串——曾导致 HuggingFace 计算出相对路径 `hub`，
   把数 GB 模型下载进项目目录。回退值必须始终非空。
2. 就绪检查误报——模型缺失却说就位，会让主程序带着坏环境启动。
"""

import os

import pytest

from services.model_readiness import (
    HF_MODELS,
    PADDLE_MODELS,
    LocalModelsNotReadyError,
    check_local_models,
    ensure_local_models,
)


def _populate_models(hf_home: str, paddlex_home: str) -> None:
    """按各自库的真实目录结构造出模型目录，供就绪检查命中。

    Args:
        hf_home: HuggingFace 缓存根目录。
        paddlex_home: PaddleX 缓存根目录。
    """
    for model_id in HF_MODELS:
        os.makedirs(
            os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
        )
    for model_name in PADDLE_MODELS:
        os.makedirs(os.path.join(paddlex_home, "official_models", model_name))


class TestCachePathResolution:
    """缓存根目录解析——回退值必须非空。"""

    def test_hf_home_unset_falls_back_to_nonempty_default(self, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)

        report = check_local_models()

        assert report.hf_home
        assert report.hf_home_from_env is False

    def test_hf_home_empty_string_falls_back_to_default(self, monkeypatch):
        """回归：空字符串曾被直接写进 HF_HOME，导致模型落到相对路径 hub/。"""
        monkeypatch.setenv("HF_HOME", "")

        report = check_local_models()

        assert report.hf_home
        assert report.hf_home_from_env is False

    def test_hf_home_whitespace_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HF_HOME", "   ")

        report = check_local_models()

        assert report.hf_home
        assert report.hf_home_from_env is False

    def test_hf_home_configured_is_used_verbatim(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        report = check_local_models()

        assert report.hf_home == str(tmp_path)
        assert report.hf_home_from_env is True

    def test_paddlex_home_empty_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", "")

        report = check_local_models()

        assert report.paddlex_home
        assert report.paddlex_home_from_env is False

    def test_paddlex_home_configured_is_used_verbatim(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))

        report = check_local_models()

        assert report.paddlex_home == str(tmp_path)
        assert report.paddlex_home_from_env is True


class TestCheckLocalModels:
    """就绪检查：全就位 / 全缺失 / 部分缺失。"""

    def test_all_models_present_reports_ready(self, monkeypatch, tmp_path):
        hf_home, paddlex_home = str(tmp_path / "hf"), str(tmp_path / "px")
        _populate_models(hf_home, paddlex_home)
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", paddlex_home)

        report = check_local_models()

        assert report.ready is True
        assert report.missing == []
        assert len(report.items) == len(HF_MODELS) + len(PADDLE_MODELS)

    def test_empty_cache_dirs_reports_all_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "px"))

        report = check_local_models()

        assert report.ready is False
        assert len(report.missing) == len(HF_MODELS) + len(PADDLE_MODELS)

    def test_partial_models_reports_only_the_missing_one(self, monkeypatch, tmp_path):
        hf_home, paddlex_home = str(tmp_path / "hf"), str(tmp_path / "px")
        _populate_models(hf_home, paddlex_home)
        os.rmdir(os.path.join(paddlex_home, "official_models", PADDLE_MODELS[0]))
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", paddlex_home)

        report = check_local_models()

        assert report.ready is False
        assert [item.name for item in report.missing] == [PADDLE_MODELS[0]]

    def test_missing_item_carries_path_for_troubleshooting(self, monkeypatch, tmp_path):
        hf_home = str(tmp_path / "hf")
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "px"))

        report = check_local_models()

        hf_item = next(i for i in report.missing if i.category == "HuggingFace")
        assert hf_item.path.startswith(hf_home)


class TestEnsureLocalModels:
    """准入检查：未就绪必须阻断，且提示要指向预装脚本。"""

    def test_raises_when_models_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "px"))

        with pytest.raises(LocalModelsNotReadyError):
            ensure_local_models()

    def test_passes_when_models_present(self, monkeypatch, tmp_path):
        hf_home, paddlex_home = str(tmp_path / "hf"), str(tmp_path / "px")
        _populate_models(hf_home, paddlex_home)
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", paddlex_home)

        ensure_local_models()

    def test_error_message_points_to_preinstall_script(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "px"))

        with pytest.raises(LocalModelsNotReadyError) as exc_info:
            ensure_local_models()

        message = str(exc_info.value)
        assert "scripts/download_models.py" in message
        assert "3.2GB" in message

    def test_error_message_lists_missing_models_and_paths(self, monkeypatch, tmp_path):
        hf_home = str(tmp_path / "hf")
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "px"))

        with pytest.raises(LocalModelsNotReadyError) as exc_info:
            ensure_local_models()

        message = str(exc_info.value)
        assert HF_MODELS[0] in message
        assert hf_home in message
