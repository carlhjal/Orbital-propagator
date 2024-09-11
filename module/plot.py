import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, coords):
        self.coords = coords

    def plot(self):
        """
        Quickly thrown together plotter. ChatGPT disclaimer!
        """
        # Unpack the trajectory points for plotting
        x, y, z = zip(*self.coords)

        # Earth parameters for plotting
        earth_radius_km = 6371  # Average Earth radius in kilometers
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xe = earth_radius_km * np.outer(np.cos(u), np.sin(v))
        ye = earth_radius_km * np.outer(np.sin(u), np.sin(v))
        ze = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Earth
        ax.plot_surface(xe, ye, ze, color='lightblue', alpha=0.5)

        # Plot trajectory
        # comment out this line to only plot starting and final position
        ax.scatter(x[1:len(x)], y[1:len(x)], z[1:len(x)], color='r', marker='o')
        
        ax.scatter(x[0], y[0], z[0], color='b', marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color='g', marker='o')
        ax.axis("equal")
        # Labels and title
        ax.set_xlabel('X km')
        ax.set_ylabel('Y km')
        ax.set_zlabel('Z km')
        ax.set_title('Orbital Trajectory around Earth')

        plt.show()