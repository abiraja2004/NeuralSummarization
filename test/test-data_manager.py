import os
import sys

if len(sys.argv)<4:
    print 'python test-data_manager.py <saved_folder> <dict.txt> <*.info>'
    exit(0)

saved_folder=sys.argv[1]
dict_file=sys.argv[2]
info_file_list=sys.argv[3:]
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

lookup_dict={}
# Load dictionary
with open(dict_file, 'r') as fopen:
    first_line=fopen.readline()
    for idx,line in enumerate(fopen):
        print '%d words loaded!\r'%idx,
        parts=line.split(' ')
        key=parts[0]
        word=' '.join(parts[1:-1])
        lookup_dict[key]=word
print ''

for idx,info_file in enumerate(info_file_list):
    print 'Loading file %s %d/%d\r'%(info_file,idx+1,len(info_file_list))
    file_name=info_file.split(os.sep)[-1]
    pure_name='.'.join(file_name.split('.')[:-1])
    output_text=''
    with open(info_file,'r') as fopen:
        first_line=fopen.readline()
        for line in fopen:
            line=line if line[-1]!='\n' else line[:-1]
            parts=line.split(',')
            for part_idx,part in enumerate(parts):
                if lookup_dict.has_key(part):
                    parts[part_idx]=lookup_dict[part]
                else:
                    print 'Warning: detect a word not listed in the dictionary: %s'%part
            sentence=' '.join(parts)+'\n'
            output_text+=sentence
    with open(saved_folder+os.sep+pure_name+'.txt','w') as fopen:
        fopen.write(output_text)
