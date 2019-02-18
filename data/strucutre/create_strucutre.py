# -*- coding: UTF-8 -*-

def create_structure(save_file_name, cou=[3, 5, 6]):
    I_begin = 1001
    J_begin = 101
    K_begin = 1
    fw = open(save_file_name, 'w')
    for i in range(cou[0]):
        i_index = I_begin + i
        for j in range(cou[1]):
            j_index = J_begin + i * cou[1] + j
            for k in range(cou[2]):
                k_index = K_begin + i * cou[1] * cou[2] + j * cou[2] + k
                fw.write(str(i_index) + ' ' + str(j_index) + ' ' + str(k_index) + '\n')
    fw.close()

if __name__ == '__main__':
    create_structure('age_structure.txt')