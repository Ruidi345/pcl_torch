import weakref
import carla
import numpy as np
import pygame

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)


class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        coll_attach_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, coll_attach_transform, attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.history.append(True)
        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)

    def reset(self):
        self.history = []

    def get_collision_history(self):
        return self.history


class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.lincoln.mkz2017"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        # if callable(on_invasion_fn):
        #     self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)


class Hero_Actor(CarlaActorBase):
    def __init__(self, world,  transform, target_velocity,
                 on_collision_fn=None, autopilot=False, on_invasion_fn=None, on_los_fn=True):
        blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = '10,0,0'  # Red
            blueprint.set_attribute('color', color)
        self.transform = transform
        actor = world.spawn_actor(blueprint, self.transform)
        self.autopilot = autopilot
        self.y_diff = 0.0

        print(f'Spawned actor \"{actor.type_id}\"')

        super().__init__(world, actor)

        self.actor.set_autopilot(self.autopilot)
        # Maintain vehicle control
        #self.world = world
        self.dt = world.dt
        self.control = carla.VehicleControl()
        # self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.actors_with_transforms = []
        self.target_velocity = target_velocity
        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        # if callable(on_invasion_fn):
        #     self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)
        # if on_los_fn:
        #     self.los_sensor = LineOfSightSensor(actor)

    def reset(self, respawn_point):
        self.collision_sensor.reset()
        # self.lane_sensor.reset()
        # self.los_sensor.reset()
        self.actor.set_transform(respawn_point)
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 0.0
        self.actor.set_autopilot(self.autopilot)

    def tick(self):
        self.actor.apply_control(self.control)

    def set_velocity(self, velocity):
        #carla09
        self.actor.set_velocity(velocity)
        #carla12
        # self.actor.set_target_velocity(velocity)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

    def get_collision_history(self):
        return self.collision_hist

    def set_y_diff(self, y_diff):
        self.y_diff = y_diff


class World():
    def __init__(self, client):
        # Set map
        #carla12
        self.world = client.load_world('Town01')
        #carla09
        # self.world = client.load_world('Town01')
        self.map = self.world.get_map()

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.map = self.get_map()
        self.actor_list = []
        self.zombie_cars = []
        self.visible_zombie_cars = []
        self.fps = 30.0
        self.dt = 0.1
        self.points_to_draw = {}
        # self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)

    def apply_settings(self, dt, synchronous_mode, no_rendering_mode):
        settings = self.world.get_settings()
        settings.no_rendering_mode = no_rendering_mode
        settings.fixed_delta_seconds = dt
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)


    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def get_carla_map(self):
        return self.map

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)

    # ===============================================================================
    # Vehicle
    # ===============================================================================


class Camera(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                  attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw, sensor_tick=0.0,):
        # 原参数有height和weight
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter
        sensor_tick = 0.0

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        # camera_bp.set_attribute("image_size_x", str(width))
        # camera_bp.set_attribute("image_size_y", str(height))
        self.image_w = camera_bp.get_attribute("image_size_x").as_int()
        self.image_h = camera_bp.get_attribute("image_size_y").as_int()
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        self.render_object = RenderObject(self.image_w, self.image_h)
        # Start camera with PyGame callback
        # self.camera.listen(lambda image: pygame_callback(image, self.renderObject)) 
        print(f'Spawned actor \"{actor.type_id}\"') 
        
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image, self.render_object))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image, randerobject):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            # print('process_camera_input')
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            randerobject.surface = pygame.surfarray.make_surface(array.swapaxes(0,1))
            self.on_recv_image(array)

            # img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            # img = img[:,:,:3]
            # img = img[:, :, ::-1]
            # randerobject.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))
            # self.on_recv_image(image)

    def destroy(self):
        super().destroy()

    def get_image_weight_height(self):
        return self.image_w, self.image_h

    def get_render_object(self):
        return self.render_object

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(image, obj):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))

class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))


# ******************* observation and action dim *******************

# 作为环境中的观察量，中心点，两边车道线，车速等等
class Box():
    def __init__(self):
        obs_dim = 50
        # for continmus
        # 这两个对应的是各个obs或者act是否有最大最小限制
        # type = np.array, shape = dim, value = True 或者False
        self.bounded_above = np.full((obs_dim), False, dtype=bool)
        self.bounded_below = np.full((obs_dim), False, dtype=bool)

        self.dtype = np.dtype('float64') # dtype('float64')

        # 和self.bounded_above类似，但是这里给的是具体限制数值
        self.high = np.full((obs_dim), np.inf)
        self.low = np.full((obs_dim), -np.inf)

        # obs或者act的维度
        self.shape = (obs_dim,)

class Discrete():
    def __init__(self, n):
        # for Discrete
        self.n = n
        self.dtype = np.dtype('int64') # dtype('int64')

# 作为环境中的action，油门，转向与刹车
class MultiDiscrete():
    def __init__(self, has_brake=False, steer_dim=20, throttle_dim=5):

        # 这里是由类Discrete组成的tuple
        # tuple:
        # 1.throttle
        # 2.steer
        # 3.brake
        if has_brake:
            self.spaces = Discrete(throttle_dim), Discrete(steer_dim), Discrete(2)
        else:
            self.spaces = Discrete(throttle_dim), Discrete(steer_dim)
            # self.spaces = Discrete(5), Discrete(20)

    def __len__(self):
        return len(self.spaces)

class EnvInfo():
    def __init__(self, has_brake=False, 
                    max_steer=1.0, 
                    min_steer=-1.0,    
                    max_throttle=1.0, 
                    min_throttle=0.0,
                    distance_avg=0.0, 
                    distance_sig=0.5,
                    ref_vel_avg=5.5, 
                    ref_vel_sig=2.0,
                    yaw_avg=0.0, 
                    yaw_sig=0.2,
                    steer_dim = 20, 
                    throttle_dim = 5):
        self.observation_space = Box()
        self.action_space = MultiDiscrete(has_brake=has_brake, 
                            steer_dim=steer_dim, 
                            throttle_dim=throttle_dim)

        self.max_steer=max_steer
        self.min_steer=min_steer
        self.max_throttle=max_throttle
        self.min_throttle=min_throttle
        self.distance_avg=distance_avg
        self.distance_sig=distance_sig
        self.ref_vel_avg=ref_vel_avg
        self.ref_vel_sig=ref_vel_sig
        self.yaw_avg=yaw_avg
        self.yaw_sig=yaw_sig

    @property
    def throttle_dim(self):
        return self.action_space.spaces[0].n

    @property
    def throttle_min(self):
        # print("self.min_throttle", self.min_throttle)
        return self.min_throttle

    @property
    def throttle_max(self):
        # print("self.max_throttle", self.max_throttle)
        return self.max_throttle

    @property
    def steer_dim(self):
        return self.action_space.spaces[1].n

    @property
    def steer_min(self):
        # print("self.min_steer", self.min_steer)
        return self.min_steer

    @property
    def steer_max(self):
        # print("self.max_steer", self.max_steer)
        return self.max_steer

    @property
    def brake_dim(self):
        if len(self.action_space.spaces) == 3:
            return self.action_space.spaces[2].n
        else:
            return None

    @property
    def brake_min(self):
        return 0.0 if len(self.action_space.spaces) == 3 else None

    @property
    def brake_max(self):
        return 1.0 if len(self.action_space.spaces) == 3 else None
# ******************* observation and action dim *******************