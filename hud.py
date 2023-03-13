
"""
Welcome to CARLA manual control.
Use ARROWS or WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    M            : toggle manual transmission
    ,/.          : gear up/down
    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle
    R            : toggle recording images to disk
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

import pygame
import datetime
import math
# from carla_gym.envs.wrapper import get_actor_display_name

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name

# ===============================================================================
# HUD
# ===============================================================================

class HUD(object):
    """
        HUD class for displaying on-screen information
    """

    def __init__(self, width, height):
        self.dim = (width, height)

        # Select a monospace font for the info panel
        fonts = [x for x in pygame.font.get_fonts() if "mono" in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font_mono = pygame.font.Font(mono, 14)

        # Use default font for everything else
        font = pygame.font.Font(pygame.font.get_default_font(), 20)

        self.notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self.show_info = True
        self.info_text = []
        self.server_clock = pygame.time.Clock()
        self.vehicle = None
        self.v_rew = None
        self.ang_rew = None
        self.track_rew = None
        self.rew_now = None
        self.ep_rew = None
        self.ep_len = None

    def set_vehicle(self, vehicle):
        self.vehicle = vehicle

    def set_info(self, module_world):
        self.v_rew = module_world.v_reward
        self.ang_rew = module_world.ang_reward
        self.track_rew = module_world.track_reward
        self.rew_now = module_world.reward
        self.ep_rew = module_world.ep_reward
        self.ep_len = module_world.steps

    def tick(self, world, clock):
        if self.show_info:
            # Get all world vehicles
            vehicles = world.get_actors().filter("vehicle.*")

            # General simulation info
            self.info_text = [
                "Server:  % 16d FPS" % self.server_fps,
                "Client:  % 16d FPS" % clock.get_fps(),
                "",
                "Map:     % 20s" % world.map.name,
                # "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
                "Number of vehicles: % 8d" % len(vehicles)
            ]

            # Show info of attached vehicle
            if self.vehicle is not None:
                # Get transform, velocity and heading
                t = self.vehicle.get_transform()
                v = self.vehicle.get_velocity()
                c = self.vehicle.get_control()
                heading = "N" if abs(t.rotation.yaw) < 89.5 else ""
                heading += "S" if abs(t.rotation.yaw) > 90.5 else ""
                heading += "E" if 179.5 > t.rotation.yaw > 0.5 else ""
                heading += "W" if -0.5 > t.rotation.yaw > -179.5 else ""

                self.info_text.extend([
                    "Vehicle: % 20s" % get_actor_display_name(self.vehicle, truncate=20),
                    "",
                    "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
                    # "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
                    "Speed:   % 15.0f m/s" % (math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
                    u"Heading:% 16.0f\N{DEGREE SIGN} % 2s" % (t.rotation.yaw, heading),
                    # f"yaw:{t.rotation.yaw:.3f}",
                    ("yaw:", t.rotation.yaw, 0.0, 180.0),
                    f"Central distance:{self.vehicle.y_diff:.3f}",
                    # ("      ", self.vehicle.y_diff, -1.75, 1.75),
                    ("      ", self.vehicle.y_diff, 0, 1.75),
                    # "Height:  % 18.0f m" % t.location.z,
                    "",
                    f"Throttle:         {c.throttle:.3f}",
                    ("         ", c.throttle, 0.0, 1.0),
                    f"Steer:            {c.steer:.3f}",
                    ("      ", c.steer, -1.0, 1.0),
                    # ("Brake:", c.brake, 0.0, 1.0),
                    # ("Reverse:", c.reverse),
                    # ("Hand brake:", c.hand_brake),
                    # ("Manual:", c.manual_gear_shift),
                    # "Gear:        %s" % {-1: "R", 0: "N"}.get(c.gear, c.gear)
                ])
            else:
                self.info_text.append("Vehicle: % 20s" % "212")

            if self.v_rew is not None:
                self.info_text.extend([
                    "",
                    "v_rew: % 15.0f" % (self.v_rew),
                    "ang_rew:   % 15.0f " % (self.ang_rew),
                    "track_rew:% 15.0f" % (self.track_rew),
                    "rew_now:% 15.0f" % (self.rew_now),
                    "ep_rew:  % 18.0f " % (self.ep_rew),
                    "eps_len:", self.ep_len,
                    "",
                    ("Throttle:", c.throttle, 0.0, 1.0),
                    ("Steer:", c.steer, -1.0, 1.0),
                    ("Brake:", c.brake, 0.0, 1.0),
                    ("Reverse:", c.reverse)
                ])
            else:
                self.info_text.append("Train Info: % 10s" % "Soft PCL 4k steps")
                # self.info_text.append("Train Info: % 10s" % "Sparse PCL right")
        # Tick notifications
        self.notifications.tick(world, clock)

    def render(self, display, extra_info=[]):
        if self.show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            self.info_text.append("")
            self.info_text.extend(extra_info)
            for item in self.info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self.font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self.notifications.render(display)
        self.help.render(display)

    def on_world_tick(self, timestamp):
        # Store info when server ticks
        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def toggle_info(self):
        self.show_info = not self.show_info

    def notification(self, text, seconds=2.0):
        self.notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self.notifications.set_text("Error: %s" % text, (255, 0, 0))


# ===============================================================================
# FadingText
# ===============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ===============================================================================
# HelpText
# ===============================================================================

class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)