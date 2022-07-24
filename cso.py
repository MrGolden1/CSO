# CSO algorithm
# Competitive swarm optimization
import numpy as np

class Particle:
    def __init__(self, pos:np.ndarray, vel:np.ndarray, fitness:float):
        self.pos = pos
        self.vel = vel
        self.fitness = fitness
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
    def __init__(self, n_particles:int, n_dim:int, bounds:np.ndarray, fitness_func, thresh=0.5):
        if n_particles % 2 != 0:
            raise ValueError("Number of particles must be even")
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.particles = []
        self.archive = {} # to avoid recalculating fitnesses
        self.thresh = thresh
        self.init_swarm()
    
    def init_swarm(self):
        for i in range(self.n_particles):
            pos = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dim)
            vel = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dim)
            selected_f = self.particles[i].get_selected_features(self.thresh)
            fitness = self.fitness_func(selected_f)
            self.particles.append(Particle(pos, vel, fitness))
            self.archive[selected_f] = fitness
    
    def check_terminate():
        pass
    
    def update_particles(self,winer:int, loser:int):
        pass
        
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
            self.update_swarm()
            self.check_terminate()
            
            