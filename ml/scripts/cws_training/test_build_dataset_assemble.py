"""Unit tests for the assemble subcommand helpers in build_dataset.py."""
import csv
import tempfile
from pathlib import Path

import pytest

from build_dataset import apply_merges, load_llm_decisions


class TestApplyMerges:
    def test_single_merge(self):
        words = ["中华", "人民", "共和国"]
        merges = [(0, 3, "中华人民共和国")]
        assert apply_merges(words, merges) == ["中华人民共和国"]

    def test_no_merges(self):
        words = ["今天", "天气", "很", "好"]
        assert apply_merges(words, []) == words

    def test_merge_at_end(self):
        words = ["我", "来自", "印", "巴"]
        merges = [(2, 4, "印巴")]
        assert apply_merges(words, merges) == ["我", "来自", "印巴"]

    def test_merge_at_start(self):
        words = ["印", "巴", "两国", "关系"]
        merges = [(0, 2, "印巴")]
        assert apply_merges(words, merges) == ["印巴", "两国", "关系"]

    def test_multiple_merges(self):
        words = ["印", "巴", "两国", "关", "系"]
        merges = [(0, 2, "印巴"), (3, 5, "关系")]
        assert apply_merges(words, merges) == ["印巴", "两国", "关系"]

    def test_empty_words(self):
        assert apply_merges([], []) == []

    def test_merge_middle(self):
        words = ["他", "是", "我", "的", "同", "行"]
        merges = [(4, 6, "同行")]
        assert apply_merges(words, merges) == ["他", "是", "我", "的", "同行"]


class TestLoadLlmDecisions:
    def _write_result_file(self, dir_path: Path, fname: str, rows: list[list[str]]):
        path = dir_path / fname
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "decision"])
            for row in rows:
                w.writerow(row)

    def test_valid_results(self, tmp_path):
        results_dir = tmp_path / "llm_results"
        results_dir.mkdir()
        self._write_result_file(results_dir, "batch_0001.tsv", [
            ["1", "merge"],
            ["2", "split"],
            ["3", "merge"],
        ])
        decisions = load_llm_decisions(results_dir, 50)
        assert decisions[("batch_0001.tsv", 1)] == "merge"
        assert decisions[("batch_0001.tsv", 2)] == "split"
        assert decisions[("batch_0001.tsv", 3)] == "merge"

    def test_missing_dir(self, tmp_path):
        decisions = load_llm_decisions(tmp_path / "nonexistent", 50)
        assert decisions == {}

    def test_malformed_rows_skipped(self, tmp_path):
        results_dir = tmp_path / "llm_results"
        results_dir.mkdir()
        self._write_result_file(results_dir, "batch_0001.tsv", [
            ["1", "merge"],
            ["2", "invalid_decision"],  # bad decision
            ["", "merge"],             # empty id
            ["4", ""],                 # empty decision
        ])
        decisions = load_llm_decisions(results_dir, 50)
        assert len(decisions) == 1
        assert decisions[("batch_0001.tsv", 1)] == "merge"

    def test_wrong_column_count(self, tmp_path):
        results_dir = tmp_path / "llm_results"
        results_dir.mkdir()
        path = results_dir / "batch_0001.tsv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("id\tdecision\n")
            f.write("1\tmerge\n")
            f.write("2\tmerge\textra_column\n")  # 3 columns
            f.write("3\n")  # 1 column
        decisions = load_llm_decisions(results_dir, 50)
        assert len(decisions) == 1
        assert ("batch_0001.tsv", 1) in decisions

    def test_case_insensitive_decision(self, tmp_path):
        results_dir = tmp_path / "llm_results"
        results_dir.mkdir()
        self._write_result_file(results_dir, "batch_0001.tsv", [
            ["1", "Merge"],
            ["2", "SPLIT"],
        ])
        decisions = load_llm_decisions(results_dir, 50)
        assert decisions[("batch_0001.tsv", 1)] == "merge"
        assert decisions[("batch_0001.tsv", 2)] == "split"

    def test_multiple_batch_files(self, tmp_path):
        results_dir = tmp_path / "llm_results"
        results_dir.mkdir()
        self._write_result_file(results_dir, "batch_0001.tsv", [["1", "merge"]])
        self._write_result_file(results_dir, "batch_0002.tsv", [["1", "split"]])
        decisions = load_llm_decisions(results_dir, 50)
        assert decisions[("batch_0001.tsv", 1)] == "merge"
        assert decisions[("batch_0002.tsv", 1)] == "split"
