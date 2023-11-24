class LearningModel:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
