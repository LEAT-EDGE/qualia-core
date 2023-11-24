import git

class Git:
    def __init__(self):
        self.repo = git.Repo(search_parent_directories=True)

    def short_hash(self):
        sha = self.repo.head.commit.hexsha
        return self.repo.git.rev_parse(sha, short=8)
