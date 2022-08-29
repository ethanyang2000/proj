class constants:
    def __init__(self) -> None:
        self.target_objects = [
            'vase_02',
            'jug04',
            'jug05',
           
        ]

        self.delta = [[-1, 0],  # go up
                    [0, -1],  # go left
                    [1, 0],  # go down
                    [0, 1]]  # go right