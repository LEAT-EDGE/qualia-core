from __future__ import annotations

import git


class Git:
    def __init__(self) -> None:
        super().__init__()
        try:
            self.repo = git.Repo(search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            self.repo = None

    @property
    def short_hash(self) -> str | None:
        if self.repo is None:
            return None

        sha = self.repo.head.commit.hexsha
        return self.repo.git.rev_parse(sha, short=8)
