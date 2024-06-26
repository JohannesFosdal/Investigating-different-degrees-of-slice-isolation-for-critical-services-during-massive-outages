
class ID_generator:
    def __init__(self):
        self.id_counters = {} 

    def generate_id(self, type, core_node):
        if (type, core_node) not in self.id_counters:
            self.id_counters[(type, core_node)] = 0
        else:
            self.id_counters[(type, core_node)] += 1

        id = int(f"{type}{core_node:02d}{self.id_counters[(type, core_node)]:04d}")
        return id
    
    def get_nodes_from_core_node(self, type, core_node):
        count = self.id_counters[(type, core_node)]
        nodeList = []
        for i in range(count + 1):
            id = int(f"{type}{core_node:02d}{i:04d}")
            nodeList.append(id)
        return nodeList

    def reset_generator(self):
        print(f'Generator reset!!')
        self.id_counters.clear()