import numpy as np
import random
import prediction.predict_stdv


class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')


def objective_function(x):
    # 定义需要优化的目标函数
    NN_Engerey = prediction.predict_stdv.StructureFilter.predict_energy(x, w1, b1, w2, b2, w3, b3)
    return NN_Engerey


def constrained_objective_function(x):
    # 带约束条件的目标函数示例
    # 在这个示例中，假设有两个约束条件：x[0] >= 0 和 x[1] <= 0
    penalty = 0
    if x[0] < 0:
        penalty += abs(x[0])
    if x[1] > 0:
        penalty += abs(x[1])
    return np.sum(x ** 2) + penalty


def adaptive_inertia_weight(iteration, max_iter):
    # 自适应惯性权重策略
    return 0.5 + 0.5 * np.cos(iteration * np.pi / max_iter)


def adaptive_weight(iteration, max_iter, initial_weight=0.9, final_weight=0.4):
    # 带有收缩因子的自适应权重策略
    delta_weight = (initial_weight - final_weight) / max_iter
    return max(initial_weight - iteration * delta_weight, final_weight)


def dynamic_environment_adaptation(iteration, max_iter):
    # 动态环境适应策略
    if iteration < max_iter / 2:
        return 0.9
    else:
        return 0.4


def improved_pso(num_particles, dim, max_iter, lower_bound, upper_bound, objective_func):
    particles = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    for iteration in range(max_iter):
        # inertia_weight = adaptive_inertia_weight(iteration, max_iter)  # 自适应权重
        # inertia_weight = adaptive_weight(iteration, max_iter)  # 带有收缩因子的自适应权重
        inertia_weight = dynamic_environment_adaptation(iteration, max_iter)  # 动态环境适应策略

        for particle in particles:
            fitness = objective_func(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)

            # 引入局部搜索策略
            if random.random() < 0.2:  # 20%的概率进行局部搜索
                perturbation = np.random.uniform(-0.1, 0.1, dim)
                new_position = particle.position + perturbation
                new_position = np.clip(new_position, lower_bound, upper_bound)
                new_fitness = objective_func(new_position)

                if new_fitness < particle.best_fitness:
                    particle.position = np.copy(new_position)
                    particle.best_fitness = new_fitness
                    particle.best_position = np.copy(new_position)

            # 更新粒子的速度和位置
            cognitive_term = random.random() * 1.5 * (particle.best_position - particle.position)
            social_term = random.random() * 1.5 * (global_best_position - particle.position)
            particle.velocity = inertia_weight * particle.velocity + cognitive_term + social_term
            particle.position = particle.position + particle.velocity
            particle.position = np.clip(particle.position, lower_bound, upper_bound)

    return global_best_position, global_best_fitness


if __name__ == "__main__":
    num_particles = 30
    dim = 2
    max_iter = 100
    lower_bound = -10
    upper_bound = 10

    best_position, best_fitness = improved_pso(num_particles, dim, max_iter, lower_bound, upper_bound,
                                               constrained_objective_function)
    print("Best Position:", best_position)
    print("Best Fitness Value:", best_fitness)
