# sourcery skip: list-comprehension
import carla
import random
import pygame
import numpy as np

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()
world.unload_map_layer(carla.MapLayer.All)   

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
traffic_manager.set_random_device_seed(0)
random.seed(0)

# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()



# Retrieve the map's spawn points
spawn_points = world.get_map().get_spawn_points()




# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))



# Control object to manage vehicle controls
class ControlObject(object):
    def __init__(self, veh):

        # Conrol parameters to store the control state
        self._vehicle = veh
        self._steer = 0
        self._throttle = False
        self._brake = False
        self._steer = None
        self._steer_cache = 0
        # A carla.VehicleControl object is needed to alter the 
        # vehicle's control state
        self._control = carla.VehicleControl()

    # Check for key press events in the PyGame window
    # and define the control state
    def parse_control(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._vehicle.set_autopilot(False)
            if event.key == pygame.K_UP:
                self._throttle = True
            if event.key == pygame.K_DOWN:
                self._brake = True
            if event.key == pygame.K_RIGHT:
                self._steer = 1
            if event.key == pygame.K_LEFT:
                self._steer = -1
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                self._throttle = False
            if event.key == pygame.K_DOWN:
                self._brake = False
                self._control.reverse = False
            if event.key == pygame.K_RIGHT:
                self._steer = None
            if event.key == pygame.K_LEFT:
                self._steer = None

    # Process the current control state, change the control parameter
    # if the key remains pressed
    def process_control(self):

        if self._throttle: 
            self._control.throttle = min(self._control.throttle + 1, 1)
            self._control.gear = 1
            self._control.brake = False
        elif not self._brake:
            self._control.throttle = 0.0

        if self._brake:
            # If the down arrow is held down when the car is stationary, switch to reverse
            v = self._vehicle.get_velocity()
            v_abs = (v.x**2 + v.y**2 + v.z**2)** 0.5
            # if self._vehicle.get_velocity().length() < 0.01 and not self._control.reverse:
            if v_abs < 0.01 and not self._control.reverse:
                self._control.brake = 0.0
                self._control.gear = 1
                self._control.reverse = True
                self._control.throttle = min(self._control.throttle + 1, 1)
            elif self._control.reverse:
                self._control.throttle = min(self._control.throttle + 1, 1)
            else:
                self._control.throttle = 0.0
                self._control.brake = min(self._control.brake + 0.3, 1)
        else:
            self._control.brake = 0.0

        if self._steer is not None:
            if self._steer == 1:
                self._steer_cache += 0.03
            if self._steer == -1:
                self._steer_cache -= 0.03
            min(0.7, max(-0.7, self._steer_cache))
            self._control.steer = round(self._steer_cache,1)
        else:
            if self._steer_cache > 0.0:
                self._steer_cache *= 0.2
            if self._steer_cache < 0.0:
                self._steer_cache *= 0.2
            if 0.01 > self._steer_cache > -0.01:
                self._steer_cache = 0.0
            self._control.steer = round(self._steer_cache,1)

        # Ápply the control parameters to the ego vehicle
        self._vehicle.apply_control(self._control)

# Randomly select a vehicle to follow with the camera
# ego_vehicle = random.choice(vehicles)
blueprint_library = world.get_blueprint_library()
model3_bp = blueprint_library.find('vehicle.tesla.model3')
model3_bp.set_attribute('color', "50, 50, 200")

cur_map = world.get_map()
ego_spwan_transform = cur_map.get_spawn_points()[86]
ego_vehicle = world.spawn_actor(model3_bp, ego_spwan_transform)

# Initialise the camera floating behind the vehicle
camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera with PyGame callback
camera.listen(lambda image: pygame_callback(image, renderObject))

# Get camera dimensions
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Instantiate objects for rendering and vehicle control
renderObject = RenderObject(image_w, image_h)
controlObject = ControlObject(ego_vehicle)


# Initialise the display
pygame.init()
gameDisplay = pygame.display.set_mode((image_w,image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
# Draw black to the display
gameDisplay.fill((0,0,0))
gameDisplay.blit(renderObject.surface, (0,0))
pygame.display.flip()


# Game loop
crashed = False

while not crashed:
    # Advance the simulation time
    world.tick()
    # Update the display
    gameDisplay.blit(renderObject.surface, (0,0))
    pygame.display.flip()
    # Process the current control state
    controlObject.process_control()
    # Collect key press events
    for event in pygame.event.get():
        # If the window is closed, break the while loop
        if event.type == pygame.QUIT:
            crashed = True

        # Parse effect of key press event on control state
        controlObject.parse_control(event)

        # if event.type == pygame.KEYUP:
        #     # TAB key switches vehicle
        #     if event.key == pygame.K_TAB:
        #         ego_vehicle.set_autopilot(True)
        #         ego_vehicle = random.choice(vehicles)
        #         # Ensure vehicle is still alive (might have been destroyed)
        #         if ego_vehicle.is_alive:
        #             # Stop and remove the camera
        #             camera.stop()
        #             camera.destroy()

        #             # Spawn new camera and attach to new vehicle
        #             controlObject = ControlObject(ego_vehicle)
        #             camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
        #             camera.listen(lambda image: pygame_callback(image, renderObject))

        #             # Update PyGame window
        #             gameDisplay.fill((0,0,0))               
        #             gameDisplay.blit(renderObject.surface, (0,0))
        #             pygame.display.flip()

# Stop camera and quit PyGame after exiting game loop
camera.stop()
pygame.quit()