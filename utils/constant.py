class constants:
    def __init__(self, task_type=None) -> None:
        if task_type == 'collect':
            self.target_objects = [
                'vase_02',
                'jug04',
                'jug05',
            ]

        self.delta = [[-1, 0],  # go up
                    [0, -1],  # go left
                    [1, 0],  # go down
                    [0, 1]]  # go right
        self.actions = [
            'navigate_to',
            'pick_up',
            'put',
            'put_in',
            'lift_up',
            'put_down',
            'navigate_with_object',
            'open',
            'close',
            'wash',
            'chop',
            'cook',
            'open_doors'
        ]
        self.magnebot_radius: float = 0.22
        self.cell_size: float = (self.magnebot_radius * 2) + 0.05
        self.nav_cell_size = 0.25



class scene_const:
    def __init__(self) -> None:
        self.room_center = {
            '1b':{
                0: [32,15],
                1:[35,7],
                2:[10,19],
                3:[21,17],
                4:[50,10],
                5:[42,8]
            }
        }

class available_actions:
    def __init__(self) -> None:
        self.data = {'camera': [0, 1, 2, 3],
        'soda can': [0, 1, 2, 3], 
        'cabinet': [0, 4, 5, 6, 7, 8], 
        'bookshelf': [0, 4, 5, 6], 
        'shelf': [0, 4, 5, 6], 
        'pliers': [0, 1, 2, 3], 
        'flashlight battery': [0, 1, 2, 3], 
        'table lamp': [0, 1, 2, 3], 
        'alarm clock': [0, 1, 2, 3], 
        'sofa': [0, 4, 5, 6], 
        'floor lamp': [0, 1, 2], 
        'vase': [0, 1, 2, 3], 
        'apple': [0, 1, 2, 3, 9, 10, 11], 
        'ipod': [0, 1, 2, 3], 
        'microwave, microwave oven': [0, 7, 8], 
        'microwave': [0, 7, 8], 
        'television set': [0, 4, 5, 6], 
        'globe': [0, 1, 2, 3], 
        'kitchen utensil': [0, 1, 2, 3, 9], 
        'trumpet': [0, 1, 2, 3], 
        'pan': [0, 1, 2, 3, 9], 
        'pot': [0, 1, 2, 3, 9], 
        'banana': [0, 1, 2, 3, 9, 10, 11], 
        'sandwich': [0, 1, 2, 3, 1, 11], 
        'dishwasher': [0, 7, 8],
        'hairbrush': [0, 1, 2, 3], 
        'candle': [0, 1, 2, 3], 
        'refrigerator': [0, 7, 8], 
        'bread': [0, 1, 2, 3, 10, 11], 
        'teakettle': [0, 1, 2, 3, 9], 
        'scissors': [0, 1, 2, 3], 
        'padlock': [0, 1, 2, 3], 
        'pen': [0, 1, 2, 3], 
        'jug': [0, 1, 2, 3], 
        'dining table': [0, 4, 5, 6], 
        'skate': [0, 1, 2, 3], 
        'spoon': [0, 1, 2, 3, 9], 
        'toothbrush': [0, 1, 2, 3], 
        'jar': [0, 1, 2, 3], 
        'backpack': [0, 1, 2, 3], 
        'baseball bat': [0, 1, 2, 3], 
        'drill': [0, 1, 2, 3], 
        'bottle': [0, 1, 2, 3], 
        'bowl': [0, 1, 2, 3, 9], 
        'wineglass': [0, 1, 2, 3, 9], 
        'cassette': [0, 1, 2, 3], 
        'calculator': [0, 1, 2, 3], 
        'golf ball': [0, 1, 2, 3], 
        'beverage': [0, 1, 2, 3], 
        'headphone': [0, 1, 2, 3], 
        'lighter': [0, 1, 2, 3], 
        'orange': [0, 1, 2, 3, 9, 10, 11], 
        'screwdriver': [0, 1, 2, 3], 
        'gas cooker': [0], 
        'coffee grinder': [0, 1, 2, 3], 
        'saltshaker, salt shaker': [0, 1, 2, 3], 
        'toaster': [0, 1, 2, 3], 
        'racquet': [0, 1, 2, 3], 
        'chocolate candy': [0, 1, 2, 3],
        'watch': [0, 1, 2, 3], 
        'coin': [0, 1, 2, 3], 
        'bag, handbag, pocketbook, purse': [0, 1, 2, 3], 
        'cookie sheet': [0, 1, 2, 3, 9], 
        'table': [0, 4, 5, 6], 
        'hammer': [0, 1, 2, 3], 
        'basket': [0, 1, 2, 3], 
        'bed': [0, 6], 
        'bench': [0, 4, 5, 6], 
        'chair': [0, 4, 5, 6], 
        'suitcase': [0, 1, 2, 3], 
        'box': [0, 1, 2, 3], 
        'sculpture': [0, 1, 2, 3], 
        'book': [0, 1, 2, 3], 
        'coffee maker': [0], 
        'cup': [0, 1, 2, 3, 9], 
        'coffee table, cocktail table': [0, 4, 5, 6], 
        'pepper mill, pepper grinder': [0, 1, 2, 3], 
        'dumbbell': [0, 1, 2, 3], 
        'painting': [0], 
        'fork': [0, 1, 2, 3, 9], 
        'picture': [0], 
        'stool': [0, 1, 2], 
        'toy': [0, 1, 2, 3], 
        'comb': [0, 1, 2, 3], 
        'printer': [0], 
        'knife': [0, 1, 2, 3, 9], 
        'clothesbrush': [0, 1, 2, 3], 
        'laptop, laptop computer': [0, 1, 2, 3], 
        'computer mouse': [0, 1, 2, 3], 
        'pencil': [0, 1, 2, 3], 
        'throw pillow': [0, 1, 2, 3], 
        'plate': [0, 1, 2, 3, 9], 
        'remote': [0, 1, 2, 3], 
        'carving fork': [0, 1, 2, 3, 9], 
        'skateboard': [0, 1, 2, 3], 
        'tea tray': [0, 1, 2, 3], 
        'trunk': [0, 4, 5, 6]}
    
    def get_actions(self, cate):
        return self.data[cate]