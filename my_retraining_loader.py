import logging
from datetime import datetime
import subprocess
import os

# 设置pythonpath环境变量，使得python可以找到models目录下的模块
cwd = os.getcwd()
pythonpath = f'{cwd}:{os.path.join(cwd,"models")}'
if 'PYTHONPATH' in os.environ:
    pythonpath = f'{pythonpath}:{os.environ["PYTHONPATH"]}'

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# 设置日志格式和日志级别，并将日志输出到logs/my_retraining_loader_{timestamp}.log文件中
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/my_retraining_loader_{timestamp}.log',
    filemode='w'
)


def main():
    # 按行读取当前目录下的my_retraining_loader_input.txt文件
    with open('my_retraining_loader_input.txt', 'r') as f:
        dirs = f.readlines()
        # 去除每行末尾的换行符
        dirs = [dir.strip() for dir in dirs]

    batch_size_val = 1024
    batch_size_opt = "--batch_size"
    from_dir_opt = "--from_dir"
    
    python_executable=subprocess.check_output(["which", "python"]).decode("utf-8").strip()
    logging.info(f"python executable: {python_executable}")

    for from_dir_val in dirs:
        command = f"{python_executable} experiments/retraining.py {batch_size_opt} {batch_size_val} {from_dir_opt} {from_dir_val}"
        env = {"PYTHONPATH": pythonpath}
        logging.info(f"command starting: {command}")
        process = subprocess.Popen(command, shell=True, env=env)
        # 等待进程结束，并判断返回值
        process.wait()
        if process.returncode != 0:
            logging.error(f"command failed: {command}")
        else:
            logging.info(f"command succeeded: {command}")


if __name__ == '__main__':
    main()
