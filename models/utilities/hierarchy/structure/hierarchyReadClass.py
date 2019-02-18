# -*- coding: UTF-8 -*-
from utilities.hierarchy.structure.nodeClass import *

class HierarchyReadClass:
    # the code of the node is int type
    def __init__(self, hierarchy_file):
        self.node_dict = {}
        self.paths_dict = {}
        self._create_hierarchy(hierarchy_file)

    def _create_hierarchy(self, hierarchy_file):
        with open(hierarchy_file) as f:
            lines = f.readlines()
        leafs_code_set = set()
        roots_code_set = set()
        inners_code_set = set()

        for line in lines:
            line = line.split()
            line = [int(a) for a in line]
            self.paths_dict[line[-1]] = line
            leafs_code_set.add(line[-1])
            roots_code_set.add(line[0])
            for code in line[0:-1]:
                inners_code_set.add(code)

            if len(line) == 1:
                code = line[0]
                if code not in self.node_dict:
                    node = NodeClass(code, 0)
                    self.node_dict[code] = node
            else:
                for i in range(len(line)):
                    code = line[i]
                    if code in self.node_dict:
                        node = self.node_dict[code]
                    else:
                        node = NodeClass(code, i)
                    # judge if it is root node
                    if i > 0:
                        parent_code = line[i - 1]
                        node.change_parent_code(parent_code)
                    # judge if it is leaf node
                    if i < len(line) - 1:
                        child_code = line[i + 1]
                        node.add_child_code(child_code)
                    # add or edit node
                    self.node_dict[code] = node

        self.leafs_code_list = sorted(list(leafs_code_set))
        self.inner_nodes_code_list = sorted(list(inners_code_set))
        self.root_nodes_code_list = sorted(list(roots_code_set))

        for code in self.inner_nodes_code_list:
            self.node_dict[code].sort_children_code()
        for code in self.root_nodes_code_list:
            self.node_dict[code].sort_children_code()

    def get_node_dict(self):
        return self.node_dict

    def get_paths_dict(self):
        return self.paths_dict

    def get_leafs_code_list(self):
        return self.leafs_code_list

    def get_inners_code_list(self):
        return self.inner_nodes_code_list

    def get_roots_code_list(self):
        return self.root_nodes_code_list

    def get_hierarchy_info(self):
        hierarchy_info = {}
        hierarchy_info['roots_code_list'] = self.get_roots_code_list()
        hierarchy_info['inners_code_list'] = self.get_inners_code_list()
        hierarchy_info['leafs_code_list'] = self.get_leafs_code_list()
        hierarchy_info['paths'] = self.get_paths_dict()
        hierarchy_info['nodes'] = self.get_node_dict()
        return hierarchy_info