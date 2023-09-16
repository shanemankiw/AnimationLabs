import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:
        
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None # (N,M,4)的ndarray, 用四元数表示的局部旋转
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)
        
        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]
        
        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:]) 
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation
    
    
    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]
    
    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass
    
    #--------------------- 你的任务 -------------------- #
    
    def decompose_rotation_with_yaxis(self, rotation):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        euler_angles = R.from_quat(rotation).as_euler("yzx")
        
        # Extract Y-axis rotation using the first component of the euler angles
        y_rotation = R.from_rotvec(np.array([0, 1, 0]) * euler_angles[0])
        y_rot = y_rotation.as_quat()

        # Obtain the XZ rotation by applying the inverse of the Y rotation to the original rotation
        xz_rotation = y_rotation.inv() * R.from_quat(rotation)
        xz_rot = xz_rotation.as_quat()

        return y_rot, xz_rot
    
    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''
        
        adjusted_data = self.raw_copy()  # Clone the original data
        
        # Adjust the root node's XZ position based on the target position
        pos_diff = target_translation_xz - adjusted_data.joint_position[frame_num, 0, [0, 2]]
        adjusted_data.joint_position[:, 0, [0, 2]] += pos_diff
        
        # Normalize the target XZ direction
        target_facing_direction_xz = target_facing_direction_xz.astype(np.float64)
        target_facing_direction_xz /= np.linalg.norm(target_facing_direction_xz)
        
        # Compute the rotation angle to align the Z-axis with the target direction
        rot_angle = np.arctan2(target_facing_direction_xz[0], target_facing_direction_xz[1])
        
        # If the rotation angle is zero, return the adjusted data
        if rot_angle == 0.0:
            return adjusted_data

        # Define the rotation around the Y-axis based on the computed angle
        rotation_transform = R.from_rotvec(np.array([0, 1, 0]) * rot_angle)
        
        # Initialize position
        initial_pos = adjusted_data.joint_position[frame_num, 0, [0, 1, 2]]
        
        # Apply rotation to all frames
        for idx in range(0, adjusted_data.joint_rotation.shape[0]):
            curr_rotation = R.from_quat(adjusted_data.joint_rotation[idx, 0])
            curr_position = adjusted_data.joint_position[idx, 0, [0, 1, 2]]
            
            # Compute new position after rotation
            new_pos_offset = rotation_transform.apply(curr_position - initial_pos)
            
            # Update rotation and position for each joint in the frame
            adjusted_data.joint_rotation[idx, 0] = (rotation_transform * curr_rotation).as_quat()
            adjusted_data.joint_position[idx, 0, [0, 2]] = np.array([new_pos_offset[0] + initial_pos[0], new_pos_offset[2] + initial_pos[2]])

        return adjusted_data

# part2
def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''
    
    res = bvh_motion1.raw_copy()
    num_frames = len(alpha)
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0

    motion1_len = float(bvh_motion1.motion_length - 1)
    motion2_len = float(bvh_motion2.motion_length - 1)


    for i in range(num_frames):
        ratio = float(i) / float(num_frames - 1)
        motion1_index = int(motion1_len * ratio)
        motion2_index = int(motion2_len * ratio)
        a = alpha[i]
        
        # Calculate the blended joint positions.
        res.joint_position[i] = (bvh_motion1.joint_position[motion1_index] * (1.0 - a)
                                 + bvh_motion2.joint_position[motion2_index] * a)
        
        for joint_i in range(bvh_motion1.joint_rotation.shape[1]):
            q0 = bvh_motion1.joint_rotation[motion1_index, joint_i]
            q1 = bvh_motion2.joint_rotation[motion2_index, joint_i]
            angle = np.arccos(np.dot(q0, q1))
            
            # Handle the case where angle is too large to avoid singularities.
            if np.abs(angle) > np.pi * 0.5:
                q0 = -q0
                
            # Linearly interpolate the quaternion values.
            q = (1.0 - a) * q0 + a * q1
            q = q / np.linalg.norm(q)
            
            res.joint_rotation[i, joint_i] = q
    
    return res

# part3
def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    
    from smooth_utils import build_loop_motion

    return res  #build_loop_motion(res)

# part4
def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    
    # Compute the starting joint's position and rotation from the first motion.
    start_position = bvh_motion1.joint_position[mix_frame1, 0, [0, 1, 2]]
    (start_Ry, _) = bvh_motion1.decompose_rotation_with_yaxis(bvh_motion1.joint_rotation[mix_frame1, 0])
    Ry = R.from_quat(start_Ry)
    start_face = Ry.apply(np.array([0, 0, 1]))
    
    # Adjust the position and orientation of the second motion.
    bvh_motion2_transformed = bvh_motion2.translation_and_rotation(0, start_position[[0, 2]], start_face[[0, 2]])
    
    # Hard concatenation.
    res.joint_position = np.concatenate([res.joint_position[:mix_frame1], bvh_motion2_transformed.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1], bvh_motion2_transformed.joint_rotation], axis=0)

    # Linear interpolation within the blending segment.
    for i in range(mix_time):
        ratio = float(i) / float(mix_time - 1)
        motion1_index = mix_frame1 + i
        motion2_index = i
        
        res.joint_position[motion1_index] = (bvh_motion1.joint_position[motion1_index] * (1.0 - ratio)
                                             + bvh_motion2_transformed.joint_position[motion2_index] * ratio)
        
        for joint_i in range(bvh_motion1.joint_rotation.shape[1]):
            q0 = bvh_motion1.joint_rotation[motion1_index, joint_i]
            q1 = bvh_motion2_transformed.joint_rotation[motion2_index, joint_i]
            angle = np.arccos(np.dot(q0, q1))
            
            # Handle the case where angle is too large to avoid singularities.
            if np.abs(angle) > np.pi * 0.5:
                q0 = -q0
                
            # Linearly interpolate the quaternion values.
            q = (1.0 - ratio) * q0 + ratio * q1
            q = q / np.linalg.norm(q)
            
            res.joint_rotation[motion1_index, joint_i] = q

    return res

