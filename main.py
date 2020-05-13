from options import Options
from solvers import create_solver

if __name__ == '__main__':
    print('[Starting]')
    opt = Options().parse()  # 读入cmd参数并初始化为Options
    solver = create_solver(opt)  # 利用配置创建求解器
    solver.run_solver()
    print('[THE END]')

