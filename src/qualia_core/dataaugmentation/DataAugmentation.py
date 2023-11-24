class DataAugmentation:
    def __init__(self, evaluate: bool = False, before: bool = False, after: bool = True) -> None:  # noqa: FBT001, FBT002
        super().__init__()
        self.evaluate = evaluate # Whether to enable data augmentation for evaluation as wel
        self.before = before
        self.after = after
