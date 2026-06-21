"""Shared project layout helpers."""

from __future__ import annotations

import os


def project_result_root(project_name: str, bucket: str) -> str:
    """Return the repo-relative result root for one project bucket.

    All generated artifacts live under the shared repo-level ``results/``
    namespace while still separating each project's training and evaluation
    outputs:

    ``results/<project_name>/train_res``
    ``results/<project_name>/eval_res``
    """
    return os.path.join("results", project_name, bucket)


def project_train_res_root(project_name: str) -> str:
    return project_result_root(project_name, "train_res")


def project_eval_res_root(project_name: str) -> str:
    return project_result_root(project_name, "eval_res")


def normalize_project_result_root(path: str | None, project_name: str, bucket: str) -> str:
    """Map default or legacy result roots onto the project-scoped location.

    We intentionally keep accepting the older bucket roots like ``train_res``
    and ``eval_res`` plus the intermediate ``res/<project>`` namespace so
    existing scripts and human habits continue to land in the new canonical
    location under ``res/<project>/<bucket>``.
    """
    if path is None:
        return project_result_root(project_name, bucket)

    normalized = os.path.normpath(str(path).strip())
    project_root = project_result_root(project_name, bucket)
    legacy_project_root = os.path.join(bucket, project_name)
    project_namespace_root = os.path.join("results", project_name)

    if normalized in {
        "",
        ".",
        "results",
        bucket,
        project_root,
        legacy_project_root,
        project_namespace_root,
    }:
        return project_root
    return path
