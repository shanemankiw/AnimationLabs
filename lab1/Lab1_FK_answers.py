import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
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


def readBVHFile(lines, i, parent, joint_name, joint_parent, joint_offset):
    tag = lines[i].split()[0] != "End"
    if tag:
        name = lines[i].split()[1]
    else:
        name = joint_name[-1] + "_end"
    joint_name.append(name)
    joint_parent.append(parent)
    nowIndex = len(joint_name)-1
    i = i + 1
    while lines[i].split()[0] != "}":
        line = lines[i].split()
        first = line[0]
        if (first == "OFFSET"):
            joint_offset.append([float(line[1]), float(line[2]), float(line[3])])
        elif (first == "JOINT" or first == "End"):
            i = readBVHFile(lines, i, nowIndex, joint_name, joint_parent, joint_offset)
        i = i + 1
    return i


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_names = []
    joint_parents = []
    joint_offsets = np.array([[0, 0, 0]])

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    # Find the ROOT joint
    root_idx = lines.index('ROOT RootJoint\n')
    # Parse the rest of the hierarchy structure
    joint_hierarchy = lines[root_idx+4:]

    # Initialize the stack with the ROOT joint
    stack = [('RootJoint', None)]
    # Pop the top joint from the stack
    joint_name, parent_name = stack.pop()

    # Add the joint to the dictionary
    joints = {}
    joints[joint_name] = parent_name

    joint_names.append('RootJoint')
    joint_parents.append(-1)

    # Find the child joints and add them to the stack
    for i in range(len(joint_hierarchy)):
        line = joint_hierarchy[i].strip()
        if (line.startswith("JOINT")) | (line.startswith("End")):
            if line.startswith("End"):
                child_name = f'{joint_name}_end'
            else:
                child_name = line.split()[1]

            joint_names.append(child_name)
            joint_parents.append(joint_names.index(joint_name))

            # if line.startswith("JOINT"):
            stack.append((child_name, joint_name))
            joint_name=child_name
        elif line.startswith("}"):
            if len(stack)==0:
                joint_name='RootJoint'
            else:
                tmp, joint_name=stack.pop()
        elif line.startswith("OFFSET"):
            parts = line.split()
            offset_values = [float(x) for x in parts[1:]]
            joint_offsets = np.append(joint_offsets, [offset_values], axis=0)
    return joint_names, joint_parents, joint_offsets


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    num = len(joint_name)

    joint_positions = np.empty((num, 3))
    joint_orientations = np.empty((num, 4))
    joint_R = []

    frame_data = motion_data[frame_id]
    frame_data = frame_data.reshape(-1, 3)

    # origin
    joint_positions[0] = joint_offset[0] + frame_data[0]
    joint_R.append(R.from_euler('XYZ', frame_data[1], degrees=True))

    frame_idx = 2
    for i in range(1, len(joint_name)):
        p = joint_parent[i]
        if joint_name[i].endswith('_end'):
            joint_R.append(R.from_euler('XYZ', [0., 0., 0.], degrees=True))
        else:
            joint_R.append(joint_R[p] * R.from_euler('XYZ', frame_data[frame_idx], degrees=True))
            frame_idx += 1
        joint_positions[i] = joint_positions[p] + joint_R[p].as_matrix() @ joint_offset[i]

    for i in range(len(joint_R)):
        joint_orientations[i] = joint_R[i].as_quat()

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)

    # print(joint_name_T)
    # print(joint_name_A)

    motion_data_A = load_motion_data(A_pose_bvh_path)
    A_dict = {}
    cnt = 1
    for i in range(len(joint_name_A)):
        if not joint_name_A[i].endswith("_end"):
            A_dict[joint_name_A[i]] = cnt
            cnt += 1

    # print(A_dict)
    motion_data = []
    for frame_id in range(len(motion_data_A)):
        frame_data_A = motion_data_A[frame_id]
        frame_data_A = frame_data_A.reshape((-1, 3))
        frame_data = []
        frame_data.append(frame_data_A[0])
        for i in range(len(joint_name_T)):
            name = joint_name_T[i]
            if not name.endswith("_end"):
                rotation = frame_data_A[A_dict[name]]
                if name == "lShoulder":
                    rotation[-1] -= 45
                elif name == "rShoulder":
                    rotation[-1] += 45
                frame_data.append(rotation)
        motion_data.append(np.asarray(frame_data).reshape(1, -1))

    motion_data = np.asarray(motion_data)
    return motion_data
