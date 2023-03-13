import copy
import enum
import math
import random
import sys
import time
import traceback

import carla
import numpy as np
import pygame
import tensorflow as tf
from six.moves import xrange
from sklearn.metrics import euclidean_distances
import h5py

from env_spec import EnvSpec
from hud import HUD
from object import Camera, EnvInfo, Hero_Actor, World

# distance_avg, distance_sig = 0, 0.5

# ref_vel_avg, ref_vel_sig = 5.5, 2.0

# yaw_avg, yaw_sig = 0, 0.2

def get_waypoint_xy(wp):
    return wp.transform.location.x , wp.transform.location.y


def gau_prob(data, avg, sig):
    coef = 1/(np.power(2 * np.pi,0.5) * sig)

    power_coef = -1/(2 * np.power(sig,2))
    gau_pow = power_coef * (np.power((data - avg), 2))

    return coef * (np.exp(gau_pow))

class Mode(object):
  autopilot = 0
  manual = 1
  train = 2
  evaluate = 3


# ************************* carla wrapper *************************

class CarlaEnv():
    def __init__(self, mode=0, 
                env_info=None,
                dt=0.1,
                max_step=400,
                direction=2):

        self.mode = mode
        render = self.mode == Mode.train
        set_autopilot = self.mode == Mode.autopilot

        # ********************* carla init *********************
        self.actor_list = []
        self.dt = 0.1
        # random.seed(3)
        # init the world
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)

        self.world = World(self.client)
        self.world.apply_settings(dt=self.dt, synchronous_mode=True, no_rendering_mode=False)
        self._world = self.world.get_carla_world()
    
        self.map = self.world.get_carla_map()

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        self.spectator = self._world.get_spectator()

        self.set_initial_velocity = True
        self.print_detail = False
        self.max_step = 400



        if direction == 0:
            self.spawn_index = [183]
        elif direction == 1:
            self.spawn_index = [57]
        elif direction == -1:
            self.spawn_index = [47] 
        else:
            self.spawn_index = list(range(len(self.map.get_spawn_points())))

        self.extra_info = []
        self.display_width = 1280
        self.display_height = 720
        self.viewer_image = self.viewer_image_buffer = None
        self.direction = direction

        # ********************* carla init *********************

        # ********************* interface  *********************

        self.env_info = env_info

        self.total = 1
        self.dones = [False] * self.total
        self.num_episodes_played = 0
        self.num_episodes_success = []
        

        # ********************* interface  *********************
        # ********************* training config *********************

        self.distance_avg = self.env_info.distance_avg 
        self.distance_sig = self.env_info.distance_sig
        self.ref_vel_avg = self.env_info.ref_vel_avg
        self.ref_vel_sig = self.env_info.ref_vel_sig
        self.yaw_avg = self.env_info.yaw_avg
        self.yaw_sig = self.env_info.yaw_sig

        self.max_step = max_step

        self.distance_table = [0.01, 1, 1.4, 2.0, 2.7, 3.8, 5.3, 7.5, 10.5, 14.7, 20.6, 28.9, 40.5]

        # ********************* training config *********************
        # ********************* actors  *********************
        ego_spwan_transform = random.choice(self.map.get_spawn_points()) # fixed scenario
        self._ego_actor = Hero_Actor(self.world, ego_spwan_transform, target_velocity=0, \
                on_collision_fn=lambda e: self._collision_data(e), autopilot=set_autopilot)

        self.ego_car = self._ego_actor.get_carla_actor()
        self.ego_info = Vehicle(self._ego_actor.get_carla_actor())
        if self.mode == Mode.autopilot:
            traffic_manager.ignore_lights_percentage(self.ego_car, 100)

        if self.mode == Mode.manual:
            camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
            self._camera = Camera(self.world, camera_init_trans, attach_to=self._ego_actor, \
                    on_recv_image=lambda e: self._set_viewer_image(e))
            # vars for manual control
            self.image_w, self.image_h = self._camera.get_image_weight_height()
            self.renderObject = self._camera.get_render_object()

        if self.mode == Mode.evaluate:
            print("Evaluating")

            camera_init_trans = carla.Transform(carla.Location(x=0, z=50), carla.Rotation(pitch=-90))
            self._camera = Camera(self.world, camera_init_trans, attach_to=self._ego_actor, \
                    on_recv_image=lambda e: self._set_viewer_image(e))
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((self.display_width, self.display_height),\
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
            self.hud = HUD(self.display_width, self.display_height)
            self.hud.set_vehicle(self._ego_actor)
            #! ?????
            self.world.on_tick(self.hud.on_world_tick)


            self.throttle = []
            self.steer = []
            self.velocity = []
            self.yaw = []
            self.center_distance = []

        # ********************* actors  *********************
        
    def step(self, action):
        """
        step() wrapper function

        Args:
            action (_type_): _description_

        Returns:
            obs: list(list(np.ndarray(obs_dim)))
            reward: tuple(float,)
            done: tuple(float,)
        """
        try:
            obs, reward, done, tt = self._step(action)
        except Exception as e:
            print(f"error occurs, {sys.exc_info()[0]}:{str(e)}, spawn point: {self.spawn_id}")
            traceback.print_exc()

            obs = self.reset()
            reward = 0
            done = True
            tt = None

        obs = [[obs]]
        reward = (float(reward), )
        done = (done, )

        return [obs, reward, done, tt]

    def reset(self):

        if self.mode == Mode.evaluate and self.nums_record !=0:
            self.record_data()

        self.invloved_junctions = {}
        self.spawn_id = random.choice(self.spawn_index)
        new_spawn_point = self.map.get_spawn_points()[self.spawn_id]

        self._ego_actor.reset(new_spawn_point)


        if self.set_initial_velocity:
            # yaw = (self._ego_car.actor.get_transform().rotation.yaw) * np.pi / 180.0
            yaw = (new_spawn_point.rotation.yaw) * np.pi / 180.0
            init_speed = carla.Vector3D(x=4.0 * np.cos(yaw),y=4.0 * np.sin(yaw))
            self._ego_actor.set_velocity(init_speed)
        self._ego_actor.tick()    # apply control          
        self.world.tick()
        self.ego_info.update()
        # print("v: {:.3f}, vx: {:.3f}, vy: {:.3f}".format(self.ego_info.v, \
        #     self.ego_info.vehicle.get_velocity().x, self.ego_info.vehicle.get_velocity().y))

        # time.sleep(3)
        self.collision_hist = []
        self.tick = 0


        # record start point
        self.start_point = new_spawn_point.location

        start_wp_pro = self.map.get_waypoint(self.ego_info.location, project_to_road=True)
        r_side_loc = self.find_lanepoint_right(start_wp_pro)
        l_side_loc = self.find_lanepoint_left(start_wp_pro)
        self.lane_bound = [r_side_loc, l_side_loc]

        obs_reset = np.zeros((50,))
        self.dones = [False] * self.total
        # print("reset.......")

        return obs_reset

    def _step(self, actions):  # sourcery skip: boolean-if-exp-identity
        """
        obs   structure: ((obs_dim, obs_type), ...)
              content:   1.throttle
                         2.brake
        act   structure: ((obs_dim, obs_type), ...)
              content:   1.ego pos, yaw, vel
                         2.waypoints pos, yaw
        """

        # time.sleep(0.05)
        self.ego_info.update()
        # print("v: {:.3f}, vx: {:.3f}, vy: {:.3f}".format(self.ego_info.v, \
        #     self.ego_info.vehicle.get_velocity().x, self.ego_info.vehicle.get_velocity().y))
        self.tick += 1

        #! action 转换
        throttle, steer, brake = self.transfer_actions(actions)

        if self.mode == Mode.evaluate:
            self.hud.tick(self.world, self.clock)
            self.viewer_image = self._get_viewer_image()

        if self.mode in [Mode.manual, Mode.autopilot]:
            self._world.tick()
            cur_control = self.ego_car.get_control()
            # print("throttle:{:.3f}, steer:{:.3f}, brake:{:.3f},".format(cur_control.throttle,\
            #     cur_control.steer, cur_control.brake))
        else:
            self._ego_actor.control.steer = steer # smaller than 0 -> left
            self._ego_actor.control.throttle = throttle
            self._ego_actor.control.brake = brake
            self.world.tick()            

        self.spectator.set_transform(carla.Transform(self.ego_info.location +
                                        carla.Location(x=0, z=50),
                                        carla.Rotation(pitch=-90)))
        # shape=(4,)
        ego_obs = np.array([self.ego_info.v, self.ego_info._yaw])

        # ********************* generate path observation *********************
        ego_wp = self.map.get_waypoint(self.ego_info.location, project_to_road=False)
        if ego_wp is None:
            ego_wp = self.map.get_waypoint(self.ego_info.location, project_to_road=False, lane_type=carla.LaneType.Any)
        # if ego_wp.is_junction and ego_wp is not None:
        if ego_wp is not None and ego_wp.is_junction:
            junc = ego_wp.get_junction()
            cur_wp_project = self.invloved_junctions[junc.id].find_projected_wp(ego_wp)
        else:
            cur_wp_project = self.map.get_waypoint(self.ego_info.location, project_to_road=True)

        self._world.debug.draw_string(cur_wp_project.transform.location, text=str(self.tick),\
                life_time=self.dt*0.1, color=carla.Color(255,225,0))

        #* 先生成navi line,收集道路id信息
        road_lane_id, junc_id = self._pre_sample(cur_wp_project)
        paths = self._generate_paths(cur_wp_project, road_lane_id)
            
        #* 提取并可视化相关路口信息
        if self.invloved_junctions:
            junc_id = list(junc_id)
            for id, junc in self.invloved_junctions.items():

                for wps in junc.exit:
                    self.world.debug.draw_point(wps.transform.location, life_time=self.dt*1.5, color=carla.Color(255,225,0))
             
            pop_junc_ids = [recorded_junc_id for recorded_junc_id in self.invloved_junctions.keys() if recorded_junc_id not in junc_id]

            for pop_junc_id in pop_junc_ids:
                self.invloved_junctions.pop(pop_junc_id)
        # ********************* generate path observation *********************

        # ********************* generate lane boundaries *********************


        obs_left_lane_bound = self.waypoint2array(self.generate_position_list(paths, 'left'))
        obs_right_lane_bound = self.waypoint2array(self.generate_position_list(paths, 'right'))

        obs = np.concatenate((ego_obs, obs_left_lane_bound))
        obs = np.concatenate((obs, obs_right_lane_bound))
        #* lane boundaries 画图
        # ********************* generate lane boundaries *********************


        # ********************* reward&done calculation *********************        

        yaw_diff = self.angle_diff(self.ego_info._yaw, cur_wp_project.transform.rotation.yaw)
        y_dif = np.hypot(self.ego_info.x - cur_wp_project.transform.location.x, self.ego_info.y - cur_wp_project.transform.location.y)
        self._ego_actor.set_y_diff(y_dif)
        reward = self._get_reward(y_dif, self.ego_info.v, yaw_diff)


        done_collision = self._collision()
        done_out_lane = self._out_lane(self.ego_info.location)
        done_max_time_step = self._max_time_step()

        if self.tick == 50 and self.start_point.distance(self.ego_info.location) < 3:
            done = True

        done = done_collision or done_max_time_step or done_out_lane
        if done:
            self.num_episodes_success.append(True if done_max_time_step else False)
            self.num_episodes_success = self.num_episodes_success[-200:]


        if self.print_detail:
            self.show_devs(cur_wp_project)
            if self._collision():
                print("end due to collision: ", self.tick)
            if self._max_time_step():
                print("end due to max time step reached")
            if self._out_lane(self.ego_info.location):
                print("end due to run out of lane!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(done, self.tick)

        # ********************* reward&done calculation *********************  

        # 设立道路两侧检测点，为判断下一帧是否越界作准备
        self.lane_bound = [self.find_lanepoint_right(cur_wp_project),\
                 self.find_lanepoint_left(cur_wp_project)]


        #* 不需要中心线点 obs.shape=(50,)
        if self.mode ==Mode.evaluate:
            self.render()
            self.throttle.append(throttle)
            self.steer.append(steer)
            self.velocity.append(self.ego_info.v)
            self.yaw.append(yaw_diff)
            self.center_distance.append(y_dif)
            # self.distance.append()
            done = False




        self.dones = [done] * self.total

        return [obs, reward, done, None]

    def _pre_sample(self, cur_ego_wp):
        # sourcery skip: collection-builtin-to-comprehension, inline-variable, merge-nested-ifs, move-assign-in-block, remove-unused-enumerate, use-fstring-for-concatenation, use-named-expression
        # self.invloved_junctions[junc.id].chosen_lane:
        # 0: road_id
        # 1: lane_id
        # 2: total s
        # 3: start waypoint
        # 4: end waypoint
        start_wp = cur_ego_wp
        sampled_distance = 0
        deceted_distance = 0.05
        interval = 1
        lanes_and_junc_length = 0
        road_lane_id = set()
        junc_id = set()

        wps2display = []

        # *车辆在junc内的情况
        if start_wp.is_junction:
            junc = start_wp.get_junction()
            junc_id.add(junc.id)
            #* carla12
            # if self.invloved_junctions[junc.id].chosen_lane[1] > 0:
            #     remain_distace_in_junc = start_wp.s - self.invloved_junctions[junc.id].chosen_lane[4].s
            #     # print("+1", "{:.3f}, {:.3f}, {:.3f}".format(remain_distace_in_junc, start_wp.s, self.invloved_junctions[junc.id].chosen_lane[4].s))
            # else:
            #     remain_distace_in_junc = self.invloved_junctions[junc.id].chosen_lane[4].s - start_wp.s
            #     # print("-1", "{:.3f}, {:.3f}, {:.3f}".format(remain_distace_in_junc, start_wp.s, self.invloved_junctions[junc.id].chosen_lane[4].s))

            #* carla09
            remain_distace_in_junc = self.invloved_junctions[junc.id].chosen_lane[2] - start_wp.s

            if remain_distace_in_junc < 0.2 and remain_distace_in_junc > -0.03:
                remain_distace_in_junc = abs(remain_distace_in_junc)
            else:
                assert remain_distace_in_junc>0, "remain_distace_in_junc < 0"
            start_wp = self.invloved_junctions[junc.id].chosen_lane[4]
            sampled_distance += remain_distace_in_junc
            wp_info = (self.invloved_junctions[junc.id].chosen_lane[0],self.invloved_junctions[junc.id].chosen_lane[1])
            road_lane_id.add(wp_info)

        while sampled_distance < 50:
            new_wps = start_wp.next(deceted_distance)
            if len(new_wps) > 1:
                #进入路口
                junc = new_wps[0].get_junction()
                
                if junc.id not in self.invloved_junctions.keys():
                    # print("len(new_wps) > 1, ", junc.id)
                    self.invloved_junctions[junc.id] = CrossRoad(junc, start_wp, self._world, self.map)
                    self.invloved_junctions[junc.id].find_possible_exit(self.direction)
                junc_id.add(junc.id)

                # 记录当前点一直到路口的距离，再加上路口选择lane的长度之和
                distance_to_cross_junc = (self._distance_to_junc(start_wp, self.invloved_junctions[junc.id]) \
                    + self.invloved_junctions[junc.id].chosen_lane[2])
                # 更新已采样距离
                sampled_distance = distance_to_cross_junc + lanes_and_junc_length
                lanes_and_junc_length += distance_to_cross_junc

                # 统计上一段的road和lane id，并为下一段的检测重置
                wp_info = (self.invloved_junctions[junc.id].chosen_lane[0],self.invloved_junctions[junc.id].chosen_lane[1])
                road_lane_id.add(wp_info)

                # 下一段延伸线reset起始点
                deceted_distance = 0.05
                start_wp = self.invloved_junctions[junc.id].chosen_lane[4]
                
            else:
                # 没有遇到路口，统计延伸点的road和lane id
                if new_wps[0].is_junction:
                    junc = new_wps[0].get_junction()
                    
                    if junc.id not in self.invloved_junctions.keys():
                        self.invloved_junctions[junc.id] = CrossRoad(junc, start_wp, self._world, self.map)
                        self.invloved_junctions[junc.id].find_possible_exit(self.direction) 
                    junc_id.add(junc.id)
                wp_info = (new_wps[0].road_id, new_wps[0].lane_id)
                road_lane_id.add(wp_info)
                sampled_distance += interval
                deceted_distance += interval
                wps2display.append(new_wps[0])


        return road_lane_id, junc_id

    def _distance_to_junc(self, start_wp, crossroad):
        max_distance = 80
        distance = 0.05
        wp_list_to_goal_junc = [start_wp]
        interval = 1
        deceted_distance = 0.05


        while distance < max_distance:
            next_wps = start_wp.next(deceted_distance)
            if len(next_wps) == 1:
                deceted_distance += interval
                distance += interval
                wp_list_to_goal_junc.append(next_wps[0])
            else:
                junc = next_wps[0].get_junction()
                if junc.id == crossroad.id:
                    eu_distance = crossroad.chosen_lane[3].transform.location.distance\
                        (wp_list_to_goal_junc[-1].transform.location)
                    distance = eu_distance + distance
                    break
                else:
                    distance += self.invloved_junctions[junc.id].chosen_lane[2]
                    wp_list_to_goal_junc.append(self.invloved_junctions[junc.id].chosen_lane[4])

                    next_wps = self.invloved_junctions[junc.id].chosen_lane[4] 
                    deceted_distance = 0.05               

        return distance

    def _extract_junction(self, paths):
        junc_invloved_wps = []
        junc_id = []
        for path in paths:
            for wp in path:
                if wp.is_junction and wp.get_junction().id not in junc_id:
                    junc_invloved_wps.append(wp)
                    junc_id.append(wp.get_junction().id)

        return junc_invloved_wps, junc_id

    def _generate_paths(self, cur_wp, road_lane_id, max_distance=50):
        paths = []
        distance_table = [0.01, 1, 1.4, 2.0, 2.7, 3.8, 5.3, 7.5, 10.5, 14.7, 20.6, 28.9, 40.5]
        wp_list = []
        #TODO navigation: end navi point location and determinate 
        #TODO close junction issues
        #TODO hand control and test
        selected_wp = None
        road_lane_id = list(road_lane_id)
        for distance in distance_table:
            wps = cur_wp.next(distance)
            if len(wps) > 1:
                # 可选点多余一个时，用road lane id筛选
                for wp in wps:
                    if (wp.road_id, wp.lane_id) in road_lane_id:
                        selected_wp = wp
                # 如果出现不在可选road lane id集合中时，选择离上个点最近的wp
                if selected_wp is None:
                    dis = [wp_list[-1].distance(wp) for wp in wps]
                    selected_wp = wps[dis.index(min(dis))]
            else:
                selected_wp = cur_wp.next(distance)[0]
            wp_list.append(selected_wp)

        return wp_list[1:]

    def waypoint2array(self, wp_loc_list):
        """
        将世界坐标系下的location序列转换成车辆坐标系下的坐标

        Args:
            wp_loc_list (_type_): list of carla.Location

        Returns:
            np.ndarray: [wp1.x, wp1.y, wp2.x, wp2.y, ...]
        """
        obs_info = [self.ego_info.transform_car(loc) for loc in wp_loc_list]
        return np.squeeze(np.array(obs_info).reshape([1, -1]), axis=0)

    def _collision(self):
        # return len(self.collision_hist) != 0 if self.tick > 5 else False
        return len(self.collision_hist) != 0 

    def _max_time_step(self):
        return self.tick == self.max_step 

    def _collision_data(self, event):
        if self.tick > 3:
            self.collision_hist.append(event)

    def _out_lane(self, ego_cur_loc):
        # new implement
        r_loc = self.lane_bound[0]
        l_loc = self.lane_bound[1]
        # for right side        
        right = np.dot(np.array([l_loc.x - r_loc.x, l_loc.y - r_loc.y]), \
            np.array([ego_cur_loc.x - r_loc.x, ego_cur_loc.y - r_loc.y]))
        # for left side        
        left = np.dot(np.array([r_loc.x - l_loc.x, r_loc.y - l_loc.y]), \
            np.array([ego_cur_loc.x - l_loc.x, ego_cur_loc.y - l_loc.y]))

        return self.tick > 5 and ((right < 0 or left < 0))

    def _get_reward(self, distance, ego_vel, ego_yaw):
        # TODO may need weight for each reward
        distance_reward = gau_prob(distance,self.distance_avg,self.distance_sig)/gau_prob(self.distance_avg,self.distance_avg,self.distance_sig)

        ref_vel_reward = gau_prob(ego_vel, self.ref_vel_avg, self.ref_vel_sig)/gau_prob(self.ref_vel_avg, self.ref_vel_avg, self.ref_vel_sig)

        yaw_reward = gau_prob(np.radians(ego_yaw), self.yaw_avg, self.yaw_sig)/gau_prob(self.yaw_avg, self.yaw_avg, self.yaw_sig)
        # if self.print_detail:
            # print("reward: d {:.3f}, v {:.3f}, y {:.3f}".format(distance_reward,ref_vel_reward,yaw_reward))
            # print("vehicle velocity: {:.3f}".format(ego_vel))
        # print("vel:{:.3f}, rew:{:.3f}, d {:.3f}, v {:.3f}, y {:.3f}".format(ego_vel, distance_reward * ref_vel_reward * yaw_reward, distance_reward,ref_vel_reward,yaw_reward))

        return distance_reward * ref_vel_reward * yaw_reward

    def __del__(self):
        # self._world.apply_settings(self.original_settings)
        print('done')

    def wp_list_extract(self, cur_lane_wp):
        wp_list = [cur_lane_wp]
        if cur_lane_wp.is_junction is False:
            wp_list = self.wp_side_extract(wp_list, cur_lane_wp, 'right')
            wp_list = self.wp_side_extract(wp_list, cur_lane_wp, 'left')
        return wp_list
        
    def wp_side_extract(self, wp_list, wp, side):
        # 以输入的wp为起点，将所有左(右)可供变向的lane(way Point)加入到list中
        if side == 'right':
            while wp.lane_change in [carla.LaneChange.Right, carla.LaneChange.Both]:
                #! get_right_lane()
                wp = wp.get_right_lane()
                wp_list.append(wp)
        if side == 'left':
            while wp.lane_change in [carla.LaneChange.Left, carla.LaneChange.Both]:
                wp = wp.get_left_lane()
                wp_list.append(wp)
        return wp_list

    def angle_diff(self, ego_yaw, lane_yaw):
        if ego_yaw < 0:
            ego_yaw = ego_yaw + 360
        if lane_yaw < 0:
            lane_yaw = lane_yaw + 360
        angle = (abs(ego_yaw - lane_yaw)) % 180
        return min(angle, 180-angle)

    def lane_display(self, legal_pos_list_left, legal_pos_list_right):
        # Legal Lane
        # 这里的wp_list中每个way point代表了car接下来的变向选择：
        # 可以变道的一侧不会被划线
        self.draw_lane_line(legal_pos_list_left, 'left')
        self.draw_lane_line(legal_pos_list_right, 'right')

    #*  通过生成的道路中心线来得到边界
    def generate_position_list(self, wp_l, side='right'):
        # sourcery skip: for-append-to-extend, remove-unnecessary-else
        if wp_l is None:
            return None
        else:
            # 对一list中的每个waypoint(在lane中心线上的)，求这条lane左右两侧的道路边界线
            # 最后的pos_list元素数量与wp_l中的数量相同
            pos_list = []
            if side == 'right':
                for wp in wp_l:
                    pos_list.append(self.find_lanepoint_right(wp))
            elif side == 'left':
                for wp in wp_l:
                    pos_list.append(self.find_lanepoint_left(wp))
            else:
                return None
        return pos_list

    def find_lanepoint_right(self, wp):
        location_drift = carla.Location(x=-np.sin(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        y=np.cos(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        z=0.2)
        return carla.Location(wp.transform.location + location_drift)

    def find_lanepoint_left(self, wp):
        location_drift = carla.Location(x=np.sin(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        y=-np.cos(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        z=0.2)
        return carla.Location(wp.transform.location + location_drift)

    def draw_lane_line(self, pos_list, side='right'):
        if pos_list:
            for i, point in enumerate(pos_list):
                if i == len(pos_list)-1:
                    break
                else:
                    self._world.debug.draw_line(pos_list[i],
                                        pos_list[i+1],
                                        thickness=0.2,
                                        color=carla.Color(255, 51, 51),
                                        life_time=0.2)  
                # self.world.debug.draw_point(point, \
                #     life_time=self.dt*1.5, color=carla.Color(238,130,205))

    def transfer_actions(self, action):
        """_summary_

        Args:
            action (list): 
                [0]: throttle
                [1]: steer
                [2]: brake

        Returns:
            float: throttle
            float: steer
            float: brake
        """
        #TODO check value type of input
        #TODO nan
        # action = tensor_action.numpy()
        def convert_discrete_to_contin(act_dis, max_limit, min_limit, dim):
            brake_interval = (max_limit - min_limit) / dim
            act_con = min_limit + brake_interval * act_dis[0]
            act_con = max(min(act_con, max_limit), min_limit)
            return act_con
        
        throttle_discrete = action[0]
        steer_discrete = action[1]
        # print(throttle_discrete, steer_discrete) 

        throttle = convert_discrete_to_contin(throttle_discrete, self.env_info.throttle_max,\
            self.env_info.throttle_min, self.env_info.throttle_dim) 
        steer = convert_discrete_to_contin(steer_discrete, self.env_info.steer_max,\
            self.env_info.steer_min, self.env_info.steer_dim) 


        if len(self.env_info.action_space) == 3:
            brake_discrete = action[2]
            brake = convert_discrete_to_contin(brake_discrete, self.env_info.brake_max,\
                self.env_info.brake_min, self.env_info.brake_dim) 
        else:
            brake = 0.0

        # print("throttle", throttle, "steer:", steer, "brake:", brake)

        return float(throttle), float(steer), float(brake)

    def render(self, mode="human"):

        # Add metrics to HUD
        self.extra_info.extend([
            # "Reward: % 19.2f" % self.reward,
            # "Total Reward: % 19.2f" % self.total_reward,
            # "",
            # # "Maneuver:        % 11s" % maneuver,
            # #"Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
            # "Distance traveled: % 7d m" % self.distance_traveled,
            # "Center deviance:   % 7.2f m" % self.distance_from_center,
            # "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            # "Avg speed:      % 7.2f km/h" % (3.6 * self.speed_accum / self.step_count),
            # "extra_info"
        ])
        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
            
    def show_devs(self, last_way_point):
        paths = []
        distance_table = [0.01, 1, 1.4, 2.0, 2.7, 3.8, 5.3, 7.5, 10.5, 14.7, 20.6, 28.9, 40.5]
        wp_list = []
        #TODO navigation: end navi point location and determinate 
        #TODO close junction issues
        #TODO hand control and test
        # self.world.debug.draw_point(wp.transform.location, life_time=self.dt*1.5, color=carla.Color(255,0,0))
        end_wps = last_way_point.next(distance_table[-1])
        deviations = []
        if len(end_wps) > 1:
            if len(set([len(last_way_point.next(distance)) for distance in distance_table])) ==3:
                paths_ = []
                for distance in distance_table:
                    wp = last_way_point.next(distance)
                    if len(wp) == 1:
                        wp_list.append(wp[0])
                    elif len(wp) == 2:
                        deviations.append(wp)
                    else:
                        three_devations = wp
                deviations = list(map(list, zip(*deviations)))

                for deviation in deviations:
                    # paths_ = wp_list + deviation
                    paths_.append(wp_list + deviation)

                for end_wp in three_devations:
                    if end_wp.transform.location.distance(paths_[0][-1].transform.location) \
                        < end_wp.transform.location.distance(paths_[1][-1].transform.location):
                        paths.append(paths_[0]+[end_wp])
                    else:
                        paths.append(paths_[1]+[end_wp])

            # generate different path
            else:
                for distance in distance_table:
                    wp = last_way_point.next(distance)
                    if len(wp) == 1:
                        wp_list.append(wp[0])
                    else:
                        deviations.append(wp)
                deviations = list(map(list, zip(*deviations)))

                # multipaths visualization
                deviation_debug = 'found {:d} devs,'
                for deviation in deviations:
                    # wp_list.extend(deviation) #? 一直在extend？
                    path = wp_list + deviation
                    paths.append(path)
                    
        else:
            # only one path
            for distance in distance_table:
                wp = last_way_point.next(distance)
                assert len(wp) == 1, "occur deviations in the middle of the path"
                wp_list.append(wp[0])

            paths.append(wp_list)

        
        for j, path in enumerate(paths):
            for i, point in enumerate(path):
                if i == len(path)-1:
                    break
                else:
                    self._world.debug.draw_line(path[i].transform.location,
                                        path[i+1].transform.location,
                                        thickness=0.2,
                                        color=rgb_list[j],
                                        life_time=0.2)  
                point_info = str(point.road_id) +","+str(point.lane_id)

        # print('num of paths generated: {:d}'.format(len(paths)))
        path_length = [len(path) for path in paths]

    @property
    def success_episode_rate(self):
        return np.sum(np.array(self.num_episodes_success)) / len(self.num_episodes_success) if \
            len(self.num_episodes_success) != 0 else 0.0

    def set_record_cfg(self, path, nums_record):
        print("set_record_cfg.....................")
        self.nums_record = nums_record
        if self.nums_record == 0:
            return 
        self.record = h5py.File(f'{path}/record.hdf5', 'w')
        self.h5py_groups = [self.record.create_group(f"episode{i}") for i in range(self.nums_record)]


    def record_data(self):
        i = self.num_episodes_played - 1
        self.h5py_groups[i].create_dataset("throttle", data=np.array(self.throttle))
        self.h5py_groups[i].create_dataset("steer", data=np.array(self.steer))
        self.h5py_groups[i].create_dataset("velocity", data=np.array(self.velocity))
        self.h5py_groups[i].create_dataset("yaw", data=np.array(self.yaw))
        self.h5py_groups[i].create_dataset("center_distance", data=np.array(self.center_distance))
        # self.h5py_groups[i].create_dataset("steps", data=np.array(self.center_distance))


        self.throttle = []
        self.steer = []
        self.velocity = []
        self.yaw = []
        self.center_distance = []

        if self.num_episodes_played == self.nums_record:
            self.record.close()
            exit()

    def _set_viewer_image(self, image):
        #TODO increase memory
        self.viewer_image_buffer = image
        # return 0


    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer
        self.viewer_image_buffer = None
        return image

    def get_one(self):
        return self.env_info

    def __len__(self):
        return 1

    def all_done(self):
        return all(self.dones)

    def seed(self, seed):
        return seed

    # TODO
    def reset_if(self, predicate=None):
        if predicate is None:
            predicate = self.dones
        
        self.num_episodes_played += sum(predicate)

        # obs, _, _, _ = self.reset()

        for pred in predicate:
            if pred == True:
                obs = self.reset()
                return [[obs]]

            else:
                return [None]
# ************************* carla wrapper *************************   
     
class Vehicle():
    
    def __init__(self, vehicle, des_vel=7, dt=0.1):
        self.vehicle = vehicle
        self.target_vel = des_vel

        self.merge_length = 0
        self.speed_max = 40
        self.acc_max = 8
        self.shape = [self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y,
                      self.vehicle.bounding_box.extent.z]
        self.steer_max = np.deg2rad(45.0)
        # self.steer_max = math.pi / 5
        self.steer_change_max = np.deg2rad(15.0)
        wb_vec = vehicle.get_physics_control().wheels[0].position - vehicle.get_physics_control().wheels[2].position
        self.wheelbase = np.sqrt(wb_vec.x**2 + wb_vec.y**2 + wb_vec.z**2)/100
        self.dt = dt

        self.x = None
        self.y = None
        self.v = None
        self.acc = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.location = None
        self.transform = None
        self.w2v = np.zeros((3,3))
        self.update()  # initialize

    def update(self):

        self.location = self.vehicle.get_location()
        self.transform = self.vehicle.get_transform()
        self.x = self.location.x
        self.y = self.location.y

        _v = self.vehicle.get_velocity()
        _acc = self.vehicle.get_acceleration()
        self.v = np.sqrt(_v.x ** 2 + _v.y ** 2)
        self.acc = np.sqrt(_acc.x ** 2 + _acc.y ** 2)
        #TODO unifiy the yaw angle
        self._yaw = self.transform.rotation.yaw
        self.roll = self.transform.rotation.roll
        self.pitch = self.transform.rotation.pitch 
        self.wb_vec = self.vehicle.get_physics_control().wheels[0].position - \
            self.vehicle.get_physics_control().wheels[2].position
        self.yaw = math.atan2(self.wb_vec.y, self.wb_vec.x)
        # 和地图初始坐标系相同
        # self.w2v[0] = np.array([np.cos(np.radians(-self._yaw)),-np.sin(np.radians(-self._yaw)),0])
        # self.w2v[1] = np.array([np.sin(np.radians(-self._yaw)),np.cos(np.radians(-self._yaw)),-0])

        # 俯视，逆时针转90度
        self.w2v[0] = np.array([np.sin(np.radians(self._yaw)),-np.cos(np.radians(self._yaw)),0])
        self.w2v[1] = np.array([np.cos(np.radians(self._yaw)),np.sin(np.radians(self._yaw)),-0])
        self.w2v[2,2] = 1
        # print(self.location)
        # print(self.x, self.y)

    def trans2veh(self, world_loc):
        pos = np.zeros((3,1))
        veh_pos = np.zeros((3,1))
        veh_pos[0,0] = self.x
        veh_pos[1,0] = self.y
        pos[0,0] = world_loc.x
        pos[1,0] = world_loc.y
        return np.matmul(self.w2v, pos)-np.matmul(self.w2v, veh_pos)
        
    def limit_input_delta(self, delta):
        if delta >= self.steer_max:
            return self.steer_max

        if delta <= -self.steer_max:
            return -self.steer_max

        return delta

    def transform_car(self,actor_loc):
        pos_ego_tran = self._rotate_car([self.x, self.y])
        pos_actor_tran = self._rotate_car([actor_loc.x, actor_loc.y])
        pos = [pos_actor_tran[0] - pos_ego_tran[0], pos_actor_tran[1] - pos_ego_tran[1]]
        return pos      

    def _rotate_car(self, pos_world):
        yaw_radians = self.yaw
        pos_tran = [pos_world[0] * np.cos(yaw_radians) + pos_world[1] * np.sin(yaw_radians),
                    -pos_world[0] * np.sin(yaw_radians) + pos_world[1] * np.cos(yaw_radians)]
        return pos_tran


class CrossRoad():
    def __init__(self, carla_junction, last_way_point, world, map):
        self.junction = carla_junction
        self.search_interval = 3
        self.id = self.junction.id
        self.world = world
        self.dt = 0.1
        self.start_wp = last_way_point
        self.lanes = []
        self.new_lanes = []
        #TODO
        self.map = map
        self.chosen_lane = None

        self.set_entry(last_way_point)
        # self.find_possible_exit(junctions)
        

    def set_entry(self, last_way_point):
        self.entry_way_point = last_way_point

    def find_possible_exit(self, direction):

        def classify_direction(dir_product):
            if dir_product < -0.5:
                direction = -1
            elif dir_product > 0.5: 
                direction = 1
            else: 
                direction = 0
            return direction

        info_format = 's:{:.3f}, {:.3f}, startx:{:.3f}, starty:{:.3f}, endx:{:.3f}, endy:{:.3f}, sroad:{:d}, slane:{:d}, eroad:{:d}, elane:{:d}'
        if self.chosen_lane == None:
            # 找从起始点到路口前最近的点，并提取这个点的road和lane id
            for i in range(100):
                # if self.start_wp.next(float(i)+0.1)[0].is_junction:
                if self.in_this_junction(self.start_wp.next(float(i)+0.05)[0]):
                    enter_junc_wp_index = i-1
                    break
            enter_junc_distance = float(enter_junc_wp_index)+0.05
            enter_wp = self.start_wp.next(enter_junc_distance)[0]
            enterwp_x , enterwp_y = get_waypoint_xy(enter_wp)
            enter_wp_info = (enter_wp.road_id, enter_wp.lane_id)
                
            # print("in the {:d} junc, enter_junc_wp_index: {:.1f}".format(self.id, enter_junc_wp_index))
            # print("the waypoint info entering junc: ", enter_wp_info)

            # 提取出路口里这么多lane，哪些是可以作为可选行驶路线
            junc_enter_exits = self.junction.get_waypoints(carla.LaneType.Driving)
            selected_index = []
            for i, junc_enter_exit in enumerate(junc_enter_exits):
                # 路口中的其中一条lane的起点的前1m的点
                enter = junc_enter_exit[0].previous(1)[0]
                # print(i, info_format.format(junc_enter_exit[0].s, junc_enter_exit[1].s,\
                #     junc_enter_exit[0].transform.location.x, junc_enter_exit[0].transform.location.y,\
                #         junc_enter_exit[1].transform.location.x, junc_enter_exit[1].transform.location.y,\
                #             junc_enter_exit[0].road_id,junc_enter_exit[0].lane_id,\
                #                 junc_enter_exit[1].road_id, jiajunc_enter_exit[1].lane_id))
                # print('all the enter in the junc: {:d}, {:d}'.format(junc_enter_exit[0].road_id, junc_enter_exit[0].lane_id))
                # print("and the corrsponding previous: {:d}, {:d}".format(enter.road_id, enter.lane_id))
                if (enter.road_id, enter.lane_id) == enter_wp_info:
                    selected_index.append(i)
            # print('the selected index: ', selected_index)
            # print('+++++++++++++++++++++++++++++++++++++')

            exit_road_ids = []
            xodr_enter_wp = []
            # 再从提取出的可行路线里，收录其road lane id和s长度
            for id, index in enumerate(selected_index):
                #* for carla12
                # s = abs(junc_enter_exits[index][0].s - junc_enter_exits[index][1].s)
                #* for carla09
                s = 0
                exit_lane_start_wp = junc_enter_exits[index][0]
                road_id, lane_id = junc_enter_exits[index][0].road_id, junc_enter_exits[index][0].lane_id
                for wp_pair in junc_enter_exits:
                    if wp_pair[0].road_id == road_id and wp_pair[0].lane_id == lane_id:
                        s = max(s, wp_pair[0].s, wp_pair[1].s)

                # 根据road lane id来筛选
                if [road_id, lane_id] not in exit_road_ids:
                    xodr_enter_wp.append([junc_enter_exits[index][0], s])
                    exit_road_ids.append([road_id, lane_id])


                # startwp_x , startwp_y = get_waypoint_xy(junc_enter_exits[index][0])
                # endwp_x , endwp_y = get_waypoint_xy(junc_enter_exits[index][1])
                
                # driving_direction = [startwp_x - enterwp_x, startwp_y - enterwp_y]
                # lane_heading = [endwp_x - enterwp_x, endwp_y - enterwp_y]
                # dir_product = np.cross(driving_direction, lane_heading)
                
                # xodr_end = self.map.get_waypoint_xodr(road_id, lane_id, 1 if lane_id>0 else s-1)
                # xodr_endwp_x , xodr_endwp_y = get_waypoint_xy(xodr_end)
                # xodr_lane_heading = [xodr_endwp_x - enterwp_x, xodr_endwp_y - enterwp_y]
                # xodr_dir_product = np.cross(driving_direction, xodr_lane_heading)
                


                # direction = classify_direction(dir_product)
                # xodr_direction = classify_direction(xodr_dir_product)

                # # print(startwp_x , startwp_y, endwp_x , endwp_y)
                # # print(driving_direction, lane_heading, dir_product, direction)
                # # print(startwp_x , startwp_y, xodr_endwp_x , xodr_endwp_y)
                # # print(driving_direction, xodr_lane_heading, xodr_dir_product, xodr_direction)
                # # print(junc_enter_exits[index][0].road_id, junc_enter_exits[index][0].lane_id, id, s)
                # # print("=====================================")
                # self.lanes.append([
                #     junc_enter_exits[index][0].road_id,  #0
                #     junc_enter_exits[index][0].lane_id,  #1
                #     s,                                   #2
                #     junc_enter_exits[index][0],          #3
                #     junc_enter_exits[index][1],          #4
                #     dir_product,                         #5
                #     xodr_direction,                           #6
                #     id])                                 #7

            # print("nums of selected:",len(xodr_enter_wp), exit_road_ids)
            def new_lanes(xodr_enter_wp, exit_road_ids):
                
                lanes = []
                for id, (begin_wp_info, road_info) in enumerate(zip(xodr_enter_wp, exit_road_ids)):
                    road_id, lane_id = road_info
                    _, s = begin_wp_info[0], begin_wp_info[1]
                    # startwp_x , startwp_y = get_waypoint_xy(begin_wp)

                    begin_wp = self.map.get_waypoint_xodr(road_id, lane_id, s-0.5 if lane_id>0 else 0.5)
                    xodr_begin_x , xodr_begin_y = get_waypoint_xy(begin_wp) 

                    xodr_out = self.map.get_waypoint_xodr(road_id, lane_id, s/2)
                    xodr_mid_x , xodr_mid_y = get_waypoint_xy(xodr_out)       

                    xodr_out = self.map.get_waypoint_xodr(road_id, lane_id, 0.5 if lane_id>0 else s-0.5)
                    xodr_out_x , xodr_out_y = get_waypoint_xy(xodr_out)  

                    driving_direction = [xodr_begin_x - enterwp_x, xodr_begin_y - enterwp_y]
                    xodr_lane_heading = [xodr_mid_x - enterwp_x, xodr_mid_y - enterwp_y]

                    xodr_dir_product = np.cross(driving_direction, xodr_lane_heading)
                    xodr_direction = classify_direction(xodr_dir_product)
                    # print('s:', s, "driving_direction", driving_direction,"xodr_lane_heading", xodr_lane_heading )
                    # print("xodr_dir_product", xodr_dir_product)
                    # print('road_id, lane_id', road_id, lane_id, "coordinate of out: ", xodr_out_x , xodr_out_y, xodr_direction)
                    # print("coordinate of out: ", xodr_out_x , xodr_out_y, xodr_direction)  
                    lanes.append([
                        begin_wp.road_id,  #0
                        begin_wp.lane_id,  #1
                        s,                                   #2
                        begin_wp,          #3
                        xodr_out,          #4
                        # xodr_dir_product,                         #5
                        xodr_direction,                           #5
                        id])                                 #6

                return lanes

            # self.new_lanes = new_lanes(xodr_enter_wp, exit_road_ids)
            self.lanes = new_lanes(xodr_enter_wp, exit_road_ids)
                
            # direction = -1
            self.chosen_lane = self.choose_exit(direction)
            # self.chosen_lane = random.choice(self.lanes)
            self.generate_exit()
            # self.generate_lane()

    # self.invloved_junctions[junc.id].chosen_lane:
    # 0: road_id
    # 1: lane_id
    # 2: total s
    # 3: start waypoint
    # 4: end waypoint
    def generate_exit(self):
        self.exit = [self.chosen_lane[3]]
        exit_lane = [self.map.get_waypoint_xodr(self.chosen_lane[0], self.chosen_lane[1], s) \
            for s in range(0, int(self.chosen_lane[2]), 1)]
        self.exit.extend(exit_lane)
        self.exit.append(self.chosen_lane[4])
        # print("nums of exit: ",len(self.exit))
        # print(self.chosen_lane[2])

        for wp in self.exit:
            self.world.debug.draw_point(wp.transform.location, life_time=self.dt*1.5, color=carla.Color(0,255,255))
        
    def choose_exit(self, direction=-1):

        if direction == 2:
            return random.choice(self.lanes)

        chosen_lane = None
        if direction not in [lane[5] for lane in self.lanes]:
            direction = 0

        for lane in self.lanes:
            if lane[5] == direction:
                chosen_lane = lane

        # assert chosen_lane is not None
        # if chosen_lane is None:
        #     print("no chosen lane")
        # else:
        #     print("having chosen lane")
        return chosen_lane


    def find_projected_wp(self, ego_wp):
        # sourcery skip: inline-immediately-returned-variable
        lane_to_ego_distance = [ego_wp.transform.location.distance(wp.transform.location) for wp in self.exit]
        project_wp = self.exit[lane_to_ego_distance.index(min(lane_to_ego_distance))]

        return project_wp


    def next_wp_in_lane(self, waypoint):
        in_junction = waypoint.is_junction
        while in_junction:
            waypoint = waypoint.next(self.search_interval)[0]
            in_junction = waypoint.is_junction

        return waypoint

    def in_this_junction(self, waypoint):
        in_junc = False
        if waypoint.is_junction:
            junc_id = waypoint.get_junction().id
            if junc_id == self.id:
                in_junc = True
        
        return in_junc

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

    



if __name__ == '__main__':
    # world = carla_init.CarlaInit()

    env_info = EnvInfo()
    try:
        env = CarlaEnv(mode=0, env_info=env_info, direction=-1)
        env_spec = EnvSpec(env.get_one())
        print('discretize_actions', env_spec.discretize_actions)
        print('obs_space', env_spec.obs_space)
        print('obs_dims', env_spec.obs_dims) #(76,)
        print('obs_types', env_spec.obs_types) #(1,)
        print('obs_info', env_spec.obs_info) 
        print('act_space', env_spec.act_space) 
        print('act_dims', env_spec.act_dims) #[400]
        print('act_types', env_spec.act_types) #[0]
        print('act_info', env_spec.act_info)
        print('combine_actions', env_spec.combine_actions)
        print('orig_act_dims', env_spec.orig_act_dims) #(5, 20, 4)
        print('orig_act_types', env_spec.orig_act_types) #(0, 0, 0)
        print('obs_dims_and_types', env_spec.obs_dims_and_types) #((76, 1),)
        print('act_dims_and_types', env_spec.act_dims_and_types) #((400, 0),)
        print('total_obs_dim', env_spec.total_obs_dim) #76
        print('total_sampling_act_dim', env_spec.total_sampling_act_dim) #400
        env.reset()
        time.sleep(1)
        dummy_actions = [np.array([0]), np.array([0]), np.array([0])]
        # for _ in range(10000):
        #     obs, reward, done, tt = env.step(dummy_actions)

        #     if done[0] == True:
        #         env.reset()

        while True:
            obs, reward, done, tt = env.step(dummy_actions)

            if done[0] == True:
                env.reset()

    except KeyboardInterrupt:
        print('Exit by user')
    finally:
        for actor in env.actor_list:
            actor.destroy()
        del env
        print("all actor destory")

        

