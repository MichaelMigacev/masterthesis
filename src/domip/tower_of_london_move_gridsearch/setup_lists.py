
def create_list_from_ints(ints, init=True):
    list = []
    for int in ints:
        if init:
            list.append("Pos" + str(int) + "Init")
        else:
            list.append("Pos" + str(int) + "Goal")
    return list

def setup_all():
    init_setup_list = ["Pos11Init", "Pos12Init", "Pos13Init", "Pos14Init", "Pos15Init", "Pos16Init",
                       "Pos21Init", "Pos22Init", "Pos23Init", "Pos24Init", "Pos25Init", "Pos26Init",
                       "Pos31Init", "Pos32Init", "Pos33Init", "Pos34Init", "Pos35Init", "Pos36Init",
                       "Pos41Init", "Pos42Init", "Pos43Init", "Pos44Init", "Pos45Init", "Pos46Init",
                       "Pos51Init", "Pos52Init", "Pos53Init", "Pos54Init", "Pos55Init", "Pos56Init",
                       "Pos61Init", "Pos62Init", "Pos63Init", "Pos64Init", "Pos65Init", "Pos66Init"]

    goal_list = ["Pos11Goal", "Pos12Goal", "Pos13Goal", "Pos14Goal", "Pos15Goal", "Pos16Goal",
                 "Pos21Goal", "Pos22Goal", "Pos23Goal", "Pos24Goal", "Pos25Goal", "Pos26Goal",
                 "Pos31Goal", "Pos32Goal", "Pos33Goal", "Pos34Goal", "Pos35Goal", "Pos36Goal",
                 "Pos41Goal", "Pos42Goal", "Pos43Goal", "Pos44Goal", "Pos45Goal", "Pos46Goal",
                 "Pos51Goal", "Pos52Goal", "Pos53Goal", "Pos54Goal", "Pos55Goal", "Pos56Goal",
                 "Pos61Goal", "Pos62Goal", "Pos63Goal", "Pos64Goal", "Pos65Goal", "Pos66Goal"]
    return init_setup_list, goal_list

def setup_standard():
    init_setup_list = create_list_from_ints([54, 42, 34, 12, 55, 16, 25, 36, 33, 53, 23, 55, 42, 54, 52, 55, 22, 34, 42, 46, 23, 43, 13, 22])

    goal_list = create_list_from_ints(
        [31, 23, 13, 65, 41, 24, 32, 15, 11, 13, 43, 15, 21, 34, 12, 35, 41, 53, 63, 25, 61, 64, 32, 55], False)
    return init_setup_list, goal_list

def setup_multiple(init=None, goal=None):
    if init is None:
        init = [54, 42]
    if goal is None:
        goal = [31, 23]
    init_setup_list = create_list_from_ints(init)
    goal_list = create_list_from_ints(goal, False)
    return init_setup_list, goal_list

def setup_single(init=54, goal=31):
    init_setup_list = create_list_from_ints([init])
    goal_list = create_list_from_ints([goal], False)
    return init_setup_list, goal_list


if __name__ == '__main__':
    print(setup_multiple([23, 45], [11, 33]))