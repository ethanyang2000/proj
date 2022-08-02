
def get_pos_and_rot(transform, id):
    for i in range(transform.get_num()):
        if transform.get_id(i) == id:
            return transform.get_position(i), transform.get_rotation(i)
    return None, None


def l2_dis(x1, x2, y1, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5