# CSO algorithm
# Competitive swarm optimization
import numpy as np
import matplotlib
import pandas as pd
# import knn from sklearn
from sklearn.neighbors import KNeighborsClassifier

def load_audit_dataset():
    # file : audit_data/audit_risk.csv
    df = pd.read_csv('audit_data/audit_risk.csv')
    # out = last column of df
    inp = df.iloc[:, :-1]
    out = df.iloc[:, -1]
    # remove LOCATION_ID column
    inp = inp.drop(columns=['LOCATION_ID'])
    return inp, out
    

class Particle:
    def __init__(self, pos:np.ndarray, vel:np.ndarray):
        self.pos = pos
        self.vel = vel
        self.fitness = 0.0
        self.selected_f = 0

    def get_selected_features(self, thresh:float):
        sel  = ""
        for i in range(len(self.pos)):
            if self.pos[i] > thresh:
                sel += "1"
            else:
                sel += "0"
        self.selected_f = sel
        return sel

    def __str__(self) -> str:
        return f"Particle: pos={self.pos}, vel={self.vel}, fitness={self.fitness}"
    
class Swarm:
    def __init__(self, n_particles:int, n_dim:int, bounds:np.ndarray, fitness_func, thresh=0.6, phi = 0.5):
        if n_particles % 2 != 0:
            raise ValueError("Number of particles must be even")
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.particles = []
        self.archive = {} # to avoid recalculating fitnesses
        self.thresh = thresh
        self.phi = phi
        self.init_swarm()
    
    def init_swarm(self):
        for i in range(self.n_particles):
            pos = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dim)
            vel = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dim)
            self.particles.append(Particle(pos, vel))
            selected_f = self.particles[i].get_selected_features(self.thresh)
            fitness = self.fitness_func(selected_f)
            self.particles[i].fitness = fitness
            self.archive[selected_f] = fitness
    
    def check_terminate(self):
        pass
    
    def mean_x_of_particles(self):
        return np.mean([particle.pos for particle in self.particles])
    
    def update_particles(self,winer:int, loser:int):
        # Rt1, Rt2, Rt3 are three randomly generated vectors within [0, 1](n_dim)
        # new velocity of loser particle: Vl = Rt1 * Vl + Rt2 * (Xw - Xl) + phi * Rt3 * (X_mean - Xl)
        r1 = np.random.uniform(0, 1, self.n_dim)
        r2 = np.random.uniform(0, 1, self.n_dim)
        r3 = np.random.uniform(0, 1, self.n_dim)
        v1 = r1 * self.particles[loser].vel + r2 * (self.particles[winer].pos - self.particles[loser].pos) + self.phi * r3 * (self.mean_x_of_particles() - self.particles[loser].pos)
        # new position of loser particle: Xl = Xl + Vl
        x1 = self.particles[loser].pos + v1
        self.particles[loser].pos = x1
        
    def update_swarm(self):
        for particle in self.particles:
            selected_f = particle.get_selected_features(self.thresh)
            if selected_f in self.archive:
                particle.fitness = self.archive[selected_f]
            else:
                fitness = self.fitness_func(selected_f)
                self.archive[selected_f] = fitness
                particle.fitness = fitness
                
        random_queue = [ np.random.randint(0, self.n_particles) for i in range(self.n_particles) ]
        while len(random_queue) > 0:
            i = random_queue.pop()
            j = random_queue.pop()
            winer_particle, loser_particle = (i,j) if self.particles[i].fitness > self.particles[j].fitness else (j,i)
            self.update_particles(winer_particle, loser_particle)
            
    def run(self, n_iterations:int):
        for i in range(n_iterations):
            print(f"Iteration {i}")
            self.update_swarm()
            self.check_terminate()

# load audit dataset
inp, out = load_audit_dataset()
# fill missing values with mean
inp.fillna(inp.mean(), inplace=True)

def KNN_fitness(selected_f:str):
    # create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    # fit KNN classifier
    selected_cols = [i for i, x in enumerate(selected_f) if x == '1']
    knn.fit(inp.iloc[:, selected_cols], out)
    # predict
    pred = knn.predict(inp.iloc[:, selected_cols])
    # return fitness
    return np.mean(pred == out)


cso_agent = Swarm(n_particles=100, n_dim=inp.shape[1], bounds=[0, 1], fitness_func=KNN_fitness, thresh=0.5)
cso_agent.run(n_iterations=100)

best_particle = max(cso_agent.particles, key=lambda x: x.fitness)
print(best_particle.selected_f)
print(best_particle.fitness)
print('The selected features are:', inp.columns[np.array(best_particle.selected_f) == '1'])