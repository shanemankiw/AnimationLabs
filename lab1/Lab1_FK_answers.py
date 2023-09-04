import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
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


def part1_calculate_T_pose(bvh_file_path):
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
    # maintain joint_name as the mother joint of the whole process
    for i in range(len(joint_hierarchy)):
        line = joint_hierarchy[i].strip()
        if (line.startswith("JOINT")) | (line.startswith("End")):
            if line.startswith("End"):
                child_name = f'{joint_name}_end'
            else:
                child_name = line.split()[1]

            joint_names.append(child_name)
            joint_parents.append(joint_names.index(joint_name))

            stack.append((child_name, joint_name)) # make it indexable in the dict
            joint_name=child_name
        elif line.startswith("}"):
            if len(stack)==0:
                joint_name='RootJoint'
            else:
                _, joint_name=stack.pop() # this joint_name end, to the parent joint
        
        elif line.startswith("OFFSET"):
            parts = line.split()
            offset_values = [float(x) for x in parts[1:]]
            joint_offsets = np.append(joint_offsets, [offset_values], axis=0)
    
    return joint_names, joint_parents, joint_offsets


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    # 初始化
    joint_positions = np.zeros((len(joint_offset), 3))
    joint_orientations = np.zeros((len(joint_offset), 4))
    idx_offset = 0
    # 遍历offset，也就是便利所有节点
    for idx, offset in enumerate(joint_offset):
        cur_joint_name = joint_name[idx]
        parent_idx = joint_parent[idx]

        if cur_joint_name.startswith('RootJoint'):
            # 根节点，多三个position数据
            joint_positions[idx] = motion_data[frame_id, :3]
            joint_orientations[idx] = R.from_euler('XYZ', motion_data[frame_id, 3:6],degrees=True).as_quat()
        elif cur_joint_name.endswith('_end'):
            q_result = joint_orientations[parent_idx] * np.concatenate(([0], offset)) * joint_orientations[parent_idx].conj()
            joint_positions[idx] = joint_positions[parent_idx]+q_result[1:]
            idx_offset += 1
        else:
            # 普通节点
            # rotation是它自己的旋转
            rotation = R.from_euler('XYZ', motion_data[frame_id, 3*(idx-idx_offset+1):3*(idx-idx_offset+2)],degrees=True).as_matrix()
            # rot_matrix_p是它父节点的朝向，因为取的已经是joint_orientations而不是single_frame_motion_data了
            rot_matrix_p=R.from_quat(joint_orientations[parent_idx]).as_matrix()
            # tmp是它自己的「朝向」
            tmp = rot_matrix_p.dot(rotation)
            # 存进朝向list
            joint_orientations[idx]=R.from_matrix(tmp).as_quat()
            # 位置，要在父节点位置的基础上，在父节点的坐标系下计算偏移
            joint_positions[idx] = joint_positions[parent_idx]+rot_matrix_p.dot(offset)

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
