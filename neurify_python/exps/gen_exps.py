from inputs import reviews
import sys

if not len(sys.argv) == 3:
    print('Needs python experiment file and config file as argument.')
    sys.exit(0)

file_path = sys.argv[1]
config_path = sys.argv[2]

out_file = 'run_exps.sh'

out_line = ''
for review in reviews:
    for eps in [0.01,0.05,0.1]:
        out_line += 'timeout 2h python3 ' + file_path + ' --config ' + config_path + ' --eps ' + str(eps) + ' --input "' + review + '"\n'
    #out_line += 'sleep 10\n' 
    out_line += '\n'

with open(out_file,'w') as f:
    f.write(out_line)
