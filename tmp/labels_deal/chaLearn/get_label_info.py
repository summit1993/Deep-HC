
def get_label_info(file_name, save_file_name):
    labels = []
    fw = open(save_file_name, 'w')
    with open(file_name) as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.split()
            label = round(float(line[1]))
            labels.append(label)
            fw.write(line[0] + ' ' + str(label) + '\n')
    print(len(set(labels)))
    print(min(labels))
    print(max(labels))
    fw.close()

def get_label_std_info(file_name, save_file_name):
    labels = []
    fw = open(save_file_name, 'w')
    with open(file_name) as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.split()
            label = round(float(line[1]))
            std = line[2]
            labels.append(label)
            fw.write(line[0] + ' ' + str(label) + ' ' + std + '\n')
    print(len(set(labels)))
    print(min(labels))
    print(max(labels))
    fw.close()

if __name__ == '__main__':
    # get_label_info('16_all_gt.txt', 'chaLearn_info.txt')
    get_label_std_info('16_all_gt.txt', 'chaLearn_info_with_std.txt')