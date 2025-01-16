from __future__ import annotations


class Git:
    def __init__(self) -> None:
        super().__init__()
        try:
            import git

            try:
                self.repo = git.Repo(search_parent_directories=True)
            except git.InvalidGitRepositoryError:
                self.repo = None
        except ImportError:
            self.repo = None

    @property
    def short_hash(self) -> str | None:
        if self.repo is None:
            return None

        sha = self.repo.head.commit.hexsha
        return self.repo.git.rev_parse(sha, short=8)
