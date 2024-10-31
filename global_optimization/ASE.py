import random
from deap import base, creator, tools, algorithms
from ase import Atoms
from ase.calculators.vasp import Vasp
import prediction.predict_stdv


# 定义DFT计算函数
def calculate_dft_energy(individual):
    # 将遗传算法中的个体转化为ASE中的原子结构
    atoms = Atoms('H2O', positions=[(0, 0, 0), (0, 0, individual[0])])

    # 设置VASP计算器（假设使用VASP进行DFT计算）
    calc = Vasp(encut=500, xc='PBE', kpts=(4, 4, 1), ispin=1, npar=4)
    atoms.set_calculator(calc)

    # 进行DFT优化
    atoms.get_potential_energy(force=True)

    # 获取能量
    energy = atoms.get_potential_energy()

    return energy


# 创建适应度类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 创建遗传算法工具箱
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)  # 定义个体的基因
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)  # 定义个体的结构
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 定义种群的结构
toolbox.register("evaluate", calculate_dft_energy)  # 使用DFT计算能量
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 定义交叉操作
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # 定义变异操作
toolbox.register("select", tools.selTournament, tournsize=3)  # 定义选择操作

# 使用GA搜索初始解
population = toolbox.population(n=50)  # 创建50个个体的种群
num_generations = 100
for generation in range(num_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)  # 交叉和变异
    fits = prediction.predict_stdv.StructureFilter.predict_energy(coordinate_matrix, w1, b1, w2, b2, w3, b3)  # 计算适应度
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))  # 选择下一代种群

# 打印最优个体和最优能量
best_individual = tools.selBest(population, k=1)[0]
print(best_individual)
print(best_individual.fitness.values[0])

# 使用ASE的DFT功能进行局部优化
optimized_energy = calculate_dft_energy(best_individual)
print(optimized_energy)
