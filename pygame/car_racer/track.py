import numpy as np
from scipy.interpolate import splprep, splev
import random
from typing import Tuple
import pygame


class Track():
    def __init__(self, size : Tuple[int, int] = (800, 700), 
                 generated : bool = False, 
                 seed : int = -1, 
                 complexity : int = 10) -> None:
        """Creates a track class, that helps manage how track looks, controlls checkpoints, car directions, etc.

        Args:
            size (tuple): size of the track image (in pixels)
            generated (bool): if true, generates random else uses from DB
            seed (int): seed for randomness of points for track path generation
            complexity (int): number of points, more points => more complex track
        """
        self.size = size
        self.generated = generated
        self.seed = seed if seed > 0 else random.randint(0, 2**32 - 1)
        self.complexity = complexity
        self._generate_path()
    
    def _find_centroid(self, points):
        x_sum, y_sum = 0, 0
        for x, y in points:
            x_sum += x
            y_sum += y
        return x_sum / len(points), y_sum / len(points)

    def _sort_points_by_angle(self, points):
        centroid = self._find_centroid(points)
        return sorted(points, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
    
    def _align_points(self, points):
        min_x = min(points, key=lambda x: x[0])[0]
        max_x = max(points, key=lambda x: x[0])[0]
        min_y = min(points, key=lambda x: x[1])[1]
        max_y = max(points, key=lambda x: x[1])[1]

        plot_width = max_x - min_x
        plot_height = max_y - min_y

        width, height = self.size
        center_x = width // 2
        center_y = height // 2

        aligned_points = [
            (
                x - min_x + (center_x - plot_width // 2), 
                y - min_y + (center_y - plot_height // 2)
            ) for x, y in points
        ]
        return aligned_points


    def _get_closed_path(self, points, num_samples=100):
        tck, u = splprep(np.array(points).T, s=0, per=True)
        u_new = np.linspace(u.min(), u.max(), num_samples)
        
        x_new, y_new = splev(u_new, tck)
        x_new = x_new + abs(np.min(x_new)) + 20 if np.min(x_new) < 5 else x_new
        y_new = y_new + abs(np.min(y_new)) + 20 if np.min(y_new) < 5 else y_new

        points_np = np.array(points)
        points_np[0, 0], points_np[0, 1]   

        sampled_points = np.column_stack((x_new, y_new))
        sampled_points = self._align_points(sampled_points)
        
        return sampled_points
        
    def _generate_path(self):
        xlim = (1, 700)
        ylim = (1, 600)

        np.random.seed(self.seed)
        random_points = np.random.uniform(low=(xlim[0], ylim[0]), high=(xlim[1], ylim[1]), size=(self.complexity, 2))
        random_points = np.array(random_points, dtype=int)
        sorted_points = self._sort_points_by_angle(random_points.tolist())
        
        sampled_points = self._get_closed_path(sorted_points, num_samples=100)
        self.points_100 = np.array(sampled_points)
        
        sampled_points = self._get_closed_path(sorted_points, num_samples=1000)
        self.points_1000 = np.array(sampled_points)
        
        self.direction_vector = [
            tuple(self.points_100[0]),
            tuple(self.points_100[1])
        ]
    def display(self, screen):
        screen.fill((150, 255, 120))
        for x, y in self.points_1000:
            pygame.draw.circle(screen, (0, 0, 0), (x, y), 26)
            
        for x, y in self.points_1000:
            pygame.draw.circle(screen, (150, 150, 150), (x, y), 25)
            
        for x, y in self.points_100:
            pygame.draw.circle(screen, (0, 0, 0), (x, y), 2)
            
        pygame.draw.circle(screen, (150, 255, 50), self.direction_vector[0], 5)
        pygame.draw.circle(screen, (150, 255, 00), self.direction_vector[1], 5)