import os
import datetime
import subprocess
import psutil
import re
import time
import numpy as np
import sys


class Experiment_record:

    def __init__(self):
        self.result_dir = './record' 
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print("----- %s folder create complete! -----" % self.result_dir)
            
        print("===== result recording! =====")

    def get_record_path(self):

        return self.result_dir

    def get_file_index(self, name):
        index = 0
        file_list = os.listdir(self.result_dir)
        for file_name in file_list:
            if file_name.split("-")[0] == name and file_name.split(".")[-1] == 'pdf':
                index += 1

        return index

    def record_data(self, file_name, data, cover=False):
        if not os.path.isfile(self.result_dir + "/"+ file_name) or cover==True:
            fi = open(self.result_dir +"/"+ file_name,"w")
            current_time = datetime.datetime.now()
            top_str = "\n"+"=="*5+str(current_time)+"=="*5+"\n"
            fi.write(top_str)
        else:
            fi = open(self.result_dir +"/"+ file_name,"a")

        fi.write(data)
        print("write record success,the information as following:\n"+data)

        fi.close()

class Cpu_limiter:

    def __init__(self, aim="cpu", target_pid = os.getpid()):
        self.result_dir = './record'
        self.pid_id = int(target_pid)
        self.cpu_info = {}
        self.limit_range_count = 0
        self.tatget_range = 100
        self.aim = aim

    def process_stop(self, process_name):
        if process_name == "self":
            p_list = [{'name':"main_procedure", 'pid':self.pid_id},
                      {'name':"experimental", 'pid':os.getpid()}]
        else:
            p_list = [p.info for p in psutil.process_iter(attrs=['pid','name']) if process_name in p.info['name']]
        if p_list:
            for p_info in p_list:
                if os.system('kill ' + str(p_info['pid'])) == 0:
                    continue
                else:
                    raise ValueError('Something warning occured while killing %s!' % process_name)
            return True
        else:
            return False

    def acquire_cpu_info(self, info_select):
        if self.cpu_info:
            return self.cpu_info[info_select]
        else:
            cmd = "lscpu"
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            p.wait()
            if p.poll() == 0:
                out, err = p.communicate()
                if len(out) == 0:
                    print("Error! The captured information is invalid!\n")
                else:
                    buf = bytes.decode(out).split('\n')
                    for info_line in buf[:-1]:
                        [info_tips, info_content] = info_line.split(':')
                        self.cpu_info[info_tips] = info_content.strip()
                    return self.cpu_info[info_select]
            else:
                print("Error to acquire cpus information!\n")

            return 1    # default single core

    def cpu_limit(self, limit_range):
        flag = self.process_stop('cpulimit')

        cmd = 'cpulimit --pid %d --limit %d --background' % (self.pid_id, limit_range)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        p.wait()
        if p.poll() == 0:
            return True
        else:
            print('Warning! Please check the cpulimit softwave!')
            return False

    def random_range(self):
        if not os.path.isfile("limit_range.npy"):
            if self.aim=="cpu":
                cpu_number = float(self.acquire_cpu_info('CPU(s)'))
                set_rate_range = (np.random.rand(self.tatget_range)*75+5) * cpu_number      # limit range: 5%~80%
                np.save(self.result_dir+"/limit_range.npy", set_rate_range)
            elif self.aim=="gpu":
                # top_use = self.capture_cpu_status()[:-1]
                top_use = 0.75
                set_rate_range = (np.random.rand(self.tatget_range)*100+14) * top_use      # limit range: 5%~80%
                np.save(self.result_dir+"/limit_range.npy", set_rate_range)
        else:
            set_rate_range = np.load("limit_range.npy")

        try:
            current_limit = set_rate_range[self.limit_range_count]
            self.limit_range_count += 1
        except:
            print("##############\nExperimental is finish!##############\n")
            self.process_stop("self")
            exit(1)
        return current_limit

    def capture_cpu_status(self):
        # cmd = " top -d 1 -n 3 -b -p %d | awk '{line[NR]=$0} END {for(i=NR-1;i<=NR;i++) print line[i]}' " % self.pid_id
        cmd = " top -d 1 -n 3 -b -p %d | awk '{line[NR]=$9} END {print line[NR]}' " % self.pid_id
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        p.wait()
        if p.poll() == 0:
            out, err = p.communicate()
            if len(out) == 0:
                print()
            else:
                cpu_used_status = re.findall(r"\d+\.?\d*",str(out))[0] + "%"
                return cpu_used_status
        else:
            print("Error to capture cpu status of this process!")
            return 0

    def main(self):
        record_handle = Experiment_record()
        cpulimit_condition_file = "cpulimit_condition.txt"
        limit_refresh_rate = 300    # 5分钟刷新频率
        while True:
            limit_rate = self.random_range()
            self.cpu_limit(limit_rate)

            print('cpulimit %d%% success work!' % limit_rate)
            cpu_used_status = self.capture_cpu_status()
            package = 'time: %f; cpulimit_set: %d%%; cpu_used_status: %s/ \n' %(time.time(), limit_rate, cpu_used_status)
            record_handle.record_data(cpulimit_condition_file, package)

            time.sleep(limit_refresh_rate)


if __name__ == "__main__":
    if sys.argv[1]=="kill":
        handle = Cpu_limiter(target_pid = sys.argv[2])
        handle.process_stop("self")
        exit(1)
    elif sys.argv[1]=="limit":
        print("\n######### Start the %s limiter! #########\n" % sys.argv[2])
        handle = Cpu_limiter(aim=sys.argv[2], target_pid = sys.argv[3])
        handle.main()

