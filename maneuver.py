import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.cm as cm


# Constants Section
# -----------------
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of Earth in kg
EARTH_RADIUS = 6371e3  # Earth's radius in meters
ORBIT_ALTITUDE = 575e3  # Orbit altitude in meters
ANORMAL_PERIOD = 20 # In minutes of simulated time, period of time when a satellite has anormal speed
PROXIMITY_THRESHOLD = 5000e3  # Threshold distance for proximity detection
PLANE_DELTA = 30 # Inclination difference between angles

SAVE_AS_GIF = False  # Set this to True to save the animation as a GIF

# Calculate the orbital radius
ORBITAL_RADIUS = EARTH_RADIUS + ORBIT_ALTITUDE  # Total radius from Earth's center in meters

# Calculate the orbital speed (v)
ORBITAL_SPEED = np.sqrt(G * M / ORBITAL_RADIUS)  # Orbital speed in m/s

# Orbital period in simulated time
ORBITAL_PERIOD_SIMULATED = 2 * np.pi * ORBITAL_RADIUS / ORBITAL_SPEED  # Orbital period in seconds (simulated time)

# Time Converter Functions
def convert_real_to_simulated(real_time_seconds, orbital_period_simulated, real_time_duration):
    """
    Converts a given real-time duration to the corresponding simulated time.
    """
    return real_time_seconds * (orbital_period_simulated / real_time_duration)

def convert_simulated_to_real(simulated_time_seconds, orbital_period_simulated, real_time_duration):
    """
    Converts a given simulated time to the corresponding real-time duration.
    """
    return simulated_time_seconds * (real_time_duration / orbital_period_simulated)

# Real time parameters
REAL_TIME_DURATION = 10.0  # REAL_TIME_DURATION seconds of real-time for one orbital period # FIXME it is coherent just when this value is set to 10

# Time Step for the Simulation
TIME_STEP_SIMULATED = ORBITAL_PERIOD_SIMULATED / 100  # Time step in the simulation to show realistic movement
TIME_STEP_REAL = convert_simulated_to_real(TIME_STEP_SIMULATED, ORBITAL_PERIOD_SIMULATED, REAL_TIME_DURATION)  # Adjusted time step for the real-time simulation

# Total simulation duration (NUM_ORBITS orbital periods)
NUM_ORBITS = 4
TOTAL_SIMULATION_TIME = NUM_ORBITS * ORBITAL_PERIOD_SIMULATED

# Print computed values for reference
print(f"Orbital Speed: {ORBITAL_SPEED:.2f} m/s")
print(f"Orbital Period (Simulated): {ORBITAL_PERIOD_SIMULATED:.2f} seconds")
print(f"Time Step (Simulated): {TIME_STEP_SIMULATED:.2f} seconds")
print(f"Time Step (Real): {TIME_STEP_REAL:.6f} seconds")
print(f"Total Simulation Time: {TOTAL_SIMULATION_TIME:.2f} seconds ({NUM_ORBITS} orbital periods)")

# Classes Section
# ---------------
class Satellite:
    def __init__(self, satellite_id, initial_angle, orbital_speed, orbital_radius, inclination, real_time_duration=10):
        self.id = satellite_id  # Satellite ID
        self.angle = initial_angle
        self.orbital_speed = orbital_speed  # Orbital speed in m/s
        self.position = np.zeros(3)
        self.inclination = inclination
        self.orbital_radius = orbital_radius

        # Calculate the angular speed to complete one orbit in the real-time duration
        self.angular_speed = (2 * np.pi) / real_time_duration  # radians per real-time second

        # Anormal speed flag and multiplier
        self.anormal_speed_active = False
        self.speed_multiplier = 1.0

        # References to neighboring satellites (within the same plane)
        self.neighbor_front = None
        self.neighbor_back = None

        # Proximity warning flag
        self.proximity_warning_active = False

    def update_position(self, time_step_real):
        # Adjust the angular speed based on the anormal speed flag
        current_angular_speed = self.angular_speed * self.speed_multiplier if self.anormal_speed_active else self.angular_speed

        # Update the angle based on the adjusted angular speed and time step
        self.angle += current_angular_speed * time_step_real

        # Ensure the angle remains within the range of 0 to 2π
        self.angle = self.angle % (2 * np.pi)

        # Calculate the new position based on the updated angle
        self.position = [
            self.orbital_radius * np.cos(self.angle) * np.cos(self.inclination),  # X position
            self.orbital_radius * np.sin(self.angle),  # Y position
            self.orbital_radius * np.cos(self.angle) * np.sin(self.inclination)   # Z position
        ]

        # Check proximity to neighbors
        self.check_proximity_to_neighbors()

    def check_proximity_to_neighbors(self):
        self.proximity_warning_active = False  # Reset the warning flag each time

        if self.neighbor_front:
            distance_to_front = np.linalg.norm(np.array(self.position) - np.array(self.neighbor_front.position))
            if distance_to_front < PROXIMITY_THRESHOLD:
                self.proximity_warning_active = True
                print(f"Satellite {self.id}: Satellite {self.neighbor_front.id} is getting close! Distance: {distance_to_front/1e3:.2f} km")

        if self.neighbor_back:
            distance_to_back = np.linalg.norm(np.array(self.position) - np.array(self.neighbor_back.position))
            if distance_to_back < PROXIMITY_THRESHOLD:
                self.proximity_warning_active = True
                print(f"Satellite {self.id}: Satellite {self.neighbor_back.id} is getting close! Distance: {distance_to_back:.2f} meters")


class OrbitalPlane:
    def __init__(self, orbital_radius, inclination, num_satellites):
        self.orbital_radius = orbital_radius
        self.inclination = inclination
        self.satellites = []
        initial_angles = np.linspace(0, 2 * np.pi, num_satellites, endpoint=False)
        for i, angle in enumerate(initial_angles):
            self.satellites.append(Satellite(satellite_id=i + 1,  # Assign an ID starting from 1
                                             initial_angle=angle,
                                             orbital_speed=ORBITAL_SPEED,
                                             orbital_radius=orbital_radius,
                                             inclination=inclination))

        # Assign neighbors (circular topology)
        for i in range(num_satellites):
            self.satellites[i].neighbor_front = self.satellites[(i + 1) % num_satellites]
            self.satellites[i].neighbor_back = self.satellites[(i - 1) % num_satellites]

    def update_positions(self, time_step_real):
        for satellite in self.satellites:
            satellite.update_position(time_step_real)



    def get_orbital_plane_points(self, steps=100):
        t = np.linspace(0, 2 * np.pi, steps)
        x_plane = self.orbital_radius * np.cos(t) * np.cos(self.inclination)
        y_plane = self.orbital_radius * np.sin(t)
        z_plane = self.orbital_radius * np.cos(t) * np.sin(self.inclination)
        return x_plane, y_plane, z_plane


class Earth:
    def __init__(self, radius, num_planes, satellites_per_plane):
        self.radius = radius
        self.orbital_planes = []
        for i in range(num_planes):
            inclination = np.deg2rad(90 - i * PLANE_DELTA)  # Different inclinations for each plane
            self.orbital_planes.append(OrbitalPlane(orbital_radius=ORBITAL_RADIUS,
                                                    inclination=inclination,
                                                    num_satellites=satellites_per_plane))

    def update_positions(self, time_step_real):
        for plane in self.orbital_planes:
            plane.update_positions(time_step_real)

# Main Function
# -------------
def main():
    # Initialize Earth with its orbital planes and satellites
    NUM_PLANES = 3
    SATELLITES_PER_PLANE = 4
    earth = Earth(radius=EARTH_RADIUS, num_planes=NUM_PLANES, satellites_per_plane=SATELLITES_PER_PLANE)

    # Initialize the satellite selection and timing
    selected_satellite = np.random.randint(SATELLITES_PER_PLANE)
    last_highlight_change_simulated = 0.0  # Last time the satellite was changed in simulated time

    # Create the plot
    fig = plt.figure(figsize=(14, 10))  # Increase the figure size, e.g., 12x12 inches
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Reduce the margins
    ax = fig.add_subplot(111, projection='3d')

    # Main loop
    def update_plot(time_step_real):
        nonlocal selected_satellite, last_highlight_change_simulated

        # Accumulate elapsed time in simulation
        if not hasattr(update_plot, "elapsed_time"):
            update_plot.elapsed_time = 0  # Initialize on the first call

        update_plot.elapsed_time += TIME_STEP_SIMULATED  # Accumulate the time step correctly with the simulated time step

        # Convert accumulated time to simulated time in minutes
        simulated_elapsed_time_minutes = update_plot.elapsed_time / 60

        # Stop updating if the total simulation time is reached
        if update_plot.elapsed_time >= TOTAL_SIMULATION_TIME:
            ani.event_source.stop()  # Stop the animation when the total simulation time is reached
            return

        ax.clear()

        # Plot Earth with more transparency and lighter color
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = earth.radius * np.outer(np.cos(u), np.sin(v))
        y = earth.radius * np.outer(np.sin(u), np.sin(v))
        z = earth.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)  # lighter color and more transparency

        # Generate colors for each plane
        colors = cm.get_cmap('tab10', len(earth.orbital_planes))  # Use a colormap with a number of colors equal to the number of planes

        # Update and plot each orbital plane and its satellites
        for i, plane in enumerate(earth.orbital_planes):
            plane.update_positions(time_step_real)
            x_plane, y_plane, z_plane = plane.get_orbital_plane_points()
            ax.plot(x_plane, y_plane, z_plane, '--', color=colors(i))

            for satellite in plane.satellites:
                # Plot the satellite as a colored circle
                ax.plot([satellite.position[0]], [satellite.position[1]], [satellite.position[2]], 'o',
                        color=colors(i), markersize=10, markerfacecolor=colors(i))

                # Plot the satellite ID inside the circle
                ax.text(satellite.position[0], satellite.position[1], satellite.position[2],
                        f'{satellite.id}', color='white', fontsize=8, ha='center', va='center', weight='bold')

                # Add an orange circle if the satellite is in proximity warning
                if satellite.proximity_warning_active:
                    ax.plot([satellite.position[0]], [satellite.position[1]], [satellite.position[2]], 'o',
                            color='orange', markersize=15, markerfacecolor='none')

            # Every ANORMAL_PERIOD minutes of simulated time, change the highlighted satellite and toggle anormal speed
            if simulated_elapsed_time_minutes - last_highlight_change_simulated >= ANORMAL_PERIOD:
                # Reset the current satellite's speed if it was in anormal mode
                earth.orbital_planes[0].satellites[selected_satellite].anormal_speed_active = False
                earth.orbital_planes[0].satellites[selected_satellite].speed_multiplier = 1.0

                # Ensure the new satellite is different from the current one
                new_selected_satellite = selected_satellite
                while new_selected_satellite == selected_satellite and SATELLITES_PER_PLANE > 1:
                    new_selected_satellite = np.random.randint(SATELLITES_PER_PLANE)
                
                selected_satellite = new_selected_satellite
                last_highlight_change_simulated = simulated_elapsed_time_minutes

                # Toggle anormal speed flag and set a random speed multiplier for the new satellite
                earth.orbital_planes[0].satellites[selected_satellite].anormal_speed_active = True
                earth.orbital_planes[0].satellites[selected_satellite].speed_multiplier = np.random.uniform(0.5, 2.0)

            # Highlight the selected satellite with a red circle
            ax.plot([earth.orbital_planes[0].satellites[selected_satellite].position[0]],
                    [earth.orbital_planes[0].satellites[selected_satellite].position[1]],
                    [earth.orbital_planes[0].satellites[selected_satellite].position[2]],
                    'ro', markersize=20, markerfacecolor='none')

        # Add a legend to the plot
        red_circle_legend = Line2D([0], [0], marker='o', color='red', markersize=15, markerfacecolor='none', linestyle='None', label='Unexpected Maneuver')
        orange_circle_legend = Line2D([0], [0], marker='o', color='orange', markersize=15, markerfacecolor='none', linestyle='None', label='Collision Danger')
        ax.legend(handles=[red_circle_legend, orange_circle_legend], loc='upper right')

        ax.set_xlim([-1.5 * ORBITAL_RADIUS, 1.5 * ORBITAL_RADIUS])
        ax.set_ylim([-1.5 * ORBITAL_RADIUS, 1.5 * ORBITAL_RADIUS])
        ax.set_zlim([-1.5 * ORBITAL_RADIUS, 1.5 * ORBITAL_RADIUS])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        ax.set_title(f"Simulated Time: {simulated_elapsed_time_minutes:.2f} minutes")

        plt.draw()

    def animate(frame):
        update_plot(TIME_STEP_REAL)

    ani = FuncAnimation(fig, animate, frames=np.linspace(0, TOTAL_SIMULATION_TIME, 200), interval=100)

    if SAVE_AS_GIF:
        # Save the animation as a GIF
        ani.save('satellite_simulation.gif', writer='pillow', fps=20)  # Adjust FPS as needed
    else:
        # Show the animation in a window
        plt.show()

if __name__ == "__main__":
    main()
