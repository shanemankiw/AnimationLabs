from answer_task1 import *
from task2_preprocess import MotionKey
import os
import pickle
from smooth_utils import decay_spring_implicit_damping_pos, decay_spring_implicit_damping_rot, quat_to_avel


class DampingParams:
    pos_diff = None
    vel_diff = None
    rot_diff = None
    avel_diff = None
    time = 0
    need_damping = False


class CharacterController():
    def __init__(self, controller) -> None:
        # static values
        self.bvh_folder_path = "motion_material/kinematic_motion"
        self.motions = []
        # self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0
        self.cur_seq = 0
        self.dampingParams = DampingParams()
        self.motion_keys = []
        # different weights for joints, positions, velocities, etc.
        self.position_weight = 1
        self.vel_weight = 1
        self.joint_position_weight = 10
        self.joint_vel_weight = 10
        self.half_life = 0.2

        # load motions and motion_keys
        for file_name in sorted(os.listdir(self.bvh_folder_path)):
            if file_name.endswith('keys'):
                with open(os.path.join(self.bvh_folder_path, file_name), 'rb') as f:
                    self.motion_keys.append(pickle.load(f))
            else:
                file_path = os.path.join(self.bvh_folder_path, file_name)
                self.motions.append(BVHMotion(file_path))

    def _find_min_cost(self, motion_key_desire, motion_key_real):
        """
        Basic motion matching, find the best match motion key in motion_key_real
        """
        pos_cost = np.sum(np.linalg.norm(
            motion_key_desire.positions - motion_key_real.positions, axis=2), axis=1)
        vel_cost = np.sum(np.linalg.norm(
            motion_key_desire.velocities - motion_key_real.velocities, axis=2), axis=1)
        joint_position_cost = np.sum(np.linalg.norm(
            motion_key_desire.joint_postions - motion_key_real.joint_postions, axis=2), axis=1)
        joint_vel_cost = np.sum(np.linalg.norm(
            motion_key_desire.joint_postions - motion_key_real.joint_postions, axis=2), axis=1)
        res = pos_cost * self.position_weight + vel_cost * self.vel_weight + \
            self.joint_position_weight * joint_position_cost + \
            self.joint_vel_weight * joint_vel_cost

        return np.argmin(res), res[np.argmin(res)]

    def update_state(self,
                     desired_pos_list,
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他

        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        
        frame_position = self.motions[self.cur_seq].joint_position[[
            self.cur_frame, self.cur_frame+1]].copy()
        frame_position[:, 0, [0, 2]] = desired_pos_list[0, [0, 2]]
        frame_rotation = self.motions[self.cur_seq].joint_rotation[[
            self.cur_frame, self.cur_frame+1]].copy()
        # apply damping
        if self.dampingParams.need_damping:
            offset, vel = decay_spring_implicit_damping_pos(
                self.dampingParams.pos_diff, self.dampingParams.vel_diff, self.half_life, self.dampingParams.time
            )
            offset_rot, avel = decay_spring_implicit_damping_rot(
                self.dampingParams.rot_diff, self.dampingParams.avel_diff, self.half_life, self.dampingParams.time
            )
            frame_position += offset
            # print(offset)
            for i in range(2):
                frame_rotation[i] = (R.from_rotvec(
                    offset_rot) * R.from_quat(frame_rotation[i])).as_quat()
            self.dampingParams.time += 1/60
        real_key = MotionKey(1)
        real_key.positions[0] = desired_pos_list
        for i in range(6):
            real_key.positions[0][5-i] -= desired_pos_list[0]
        real_key.velocities[0] = desired_vel_list
        real_key.tracking_joints = self.motion_keys[0].tracking_joints
        joint_translation, joint_orientation = self.motions[self.cur_seq].batch_forward_kinematics(
            frame_position, frame_rotation)
        for k in range(len(real_key.tracking_joints)):
            joint = real_key.tracking_joints[k]
            idx = self.motions[self.cur_seq].joint_name.index(joint)
            real_key.joint_postions[0][k] = joint_translation[0,
                                                              idx, :] - joint_translation[0, 0, :]
            real_key.joint_velocities[0][k] = (
                joint_translation[1, idx, :] - joint_translation[0, idx, :]) * 60

        min_cost = 1e20
        min_seq_id = -1
        min_frame_id = -1
        for i in range(len(self.motion_keys)):
            motion_key = self.motion_keys[i]
            cur_id, cur_cost = self._find_min_cost(motion_key, real_key)
            if cur_cost < min_cost:
                min_cost = cur_cost
                min_seq_id = i
                min_frame_id = cur_id

        if min_seq_id == self.cur_seq and abs(min_frame_id-self.cur_frame) < 10:
            self.cur_frame = (
                self.cur_frame + 1) % self.motions[0].motion_length
        else:
            # calc dampParams
            self.dampingParams.time = 0
            joint_position_dst = self.motions[min_seq_id].joint_position[[
                min_frame_id, min_frame_id+1]].copy()
            joint_rotation_dst = self.motions[min_seq_id].joint_rotation[[
                min_frame_id, min_frame_id+1]].copy()
            self.dampingParams.pos_diff = (
                frame_position[0] - joint_position_dst[0])
            self.dampingParams.pos_diff[0] = np.array([0, 0, 0])
            vel_src = (frame_position[1] - frame_position[0]) * 60
            vel_dst = (joint_position_dst[1] - joint_position_dst[0]) * 60
            self.dampingParams.vel_diff = vel_src - vel_dst

            avel_dst = quat_to_avel(joint_rotation_dst, 1/60)
            avel_src = quat_to_avel(frame_rotation, 1/60)
            self.dampingParams.rot_diff = (R.from_quat(
                frame_rotation[0]) * R.from_quat(joint_rotation_dst[0].copy()).inv()).as_rotvec()
            self.dampingParams.avel_diff = avel_src[0] - avel_dst[0]
            self.cur_seq = min_seq_id
            self.cur_frame = min_frame_id
            self.dampingParams.need_damping = True

        joint_name = self.motions[self.cur_seq].joint_name

        translation_offset = desired_pos_list[0] - joint_translation[0][0]
        translation_offset[1] = 0
        joint_translation = joint_translation[0] + translation_offset
        joint_orientation = joint_orientation[0]
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        # print(self.cur_frame, self.cur_seq)
        return joint_name, joint_translation, joint_orientation

    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''

        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)

        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.
