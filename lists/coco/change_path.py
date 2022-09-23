with open('val_data_list.txt','r') as f:
    lines = f.readlines()
new_file = open('my_val_data_list.txt', 'w+')


for line in lines:
    list = line.split()
    list[0] = 'images/val2014/' + list[0].split('/')[-1]
    list[1] = 'images_mask_gray/val2014/' + list[1].split('/')[-1]
    line = ' '.join(list)
    new_file.write(line)
    new_file.write('\n')




