import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
import task2_inverse_kinematics

def rotation_matrix(a, b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    n = np.cross(a, b)
    # 计算夹角
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(n)
    theta = np.arctan2(sin_theta, cos_theta)
    # 构造旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    rotation_matrix = np.array([[n[0]*n[0]*v+c, n[0]*n[1]*v-n[2]*s, n[0]*n[2]*v+n[1]*s],
                                 [n[0]*n[1]*v+n[2]*s, n[1]*n[1]*v+c, n[1]*n[2]*v-n[0]*s],
                                 [n[0]*n[2]*v-n[1]*s, n[1]*n[2]*v+n[0]*s, n[2]*n[2]*v+c]])
    return rotation_matrix

def get_joint_rotations(joint_name, joint_orientations, joint_parent):
    joint_rotations = np.empty(joint_orientations.shape)
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
        else:
            joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
    return joint_rotations

def get_joint_offsets(joint_name, joint_positions, joint_parent, joint_initial_position):
    joint_offsets = np.empty(joint_positions.shape)
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            joint_offsets[i] = np.array([0.,0.,0.])
        else:
            joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
    return joint_offsets

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint # the joint that does not move.
    end_joint = meta_data.end_joint # the joint to the target position

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    """
    path: root -> pelvis -> end
    path1: end -> pelvis
    path2: root -> pelvis
    """
    # path1 is the path from the end_joint to pelvis
    # path2 is the path from the root_joint to pelvis, unmoved
    # Link them together would be the complete link from end to root.
    
    # If the root_joint is the pelvis, then the path2 would be empty.
    if len(path2) == 1 and path2[0] != 0:
        path2 = []

    joint_rotations = get_joint_rotations(joint_name, joint_orientations, joint_parent)
    joint_offsets = get_joint_offsets(joint_name, joint_positions, joint_parent, joint_initial_position)

    rotation_chain = np.empty((len(path),), dtype=object)
    position_chain = np.empty((len(path), 3))
    orientation_chain = np.empty((len(path),), dtype=object)
    offset_chain = np.empty((len(path), 3))

    # 对chain进行初始化
    if len(path2) > 1:
        orientation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv()
    else:
        orientation_chain[0] = R.from_quat(joint_orientations[path[0]])

    position_chain[0] = joint_positions[path[0]]
    rotation_chain[0] = orientation_chain[0]
    offset_chain[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        # loop in the full chain
        index = path[i] # joint index
        position_chain[i] = joint_positions[index] # joint position
        if index in path2:
            # if the joint inside root_joint to pelvis
            # orientation from the next joint in the path.
            orientation_chain[i] = R.from_quat(joint_orientations[path[i + 1]])
            rotation_chain[i] = R.from_quat(joint_rotations[path[i]]).inv()
            offset_chain[i] = -joint_offsets[path[i - 1]]
        else:
            orientation_chain[i] = R.from_quat(joint_orientations[index])
            rotation_chain[i] = R.from_quat(joint_rotations[index])
            offset_chain[i] = joint_offsets[index]


    # CCD IK
    times = 10
    distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))
    end = False
    while times > 0 and distance > 1e-3 and not end:
        times -= 1
        for i in range(len(path) - 2, -1, -1):
            if joint_parent[path[i]] == -1:
                continue
            cur_pos = position_chain[i]
            c2t = target_pose - cur_pos
            c2e = position_chain[-1] - cur_pos
            axis = np.cross(c2e, c2t)
            axis = axis / np.linalg.norm(axis)
            cos = min(np.dot(c2e, c2t) / (np.linalg.norm(c2e) * np.linalg.norm(c2t)), 1.0)
            theta = np.arccos(cos)
            if theta < 1e-4:
                continue
            delta_rotation = R.from_rotvec(theta * axis)
            orientation_chain[i] = delta_rotation * orientation_chain[i]
            rotation_chain[i] = orientation_chain[i - 1].inv() * orientation_chain[i]
            for j in range(i + 1, len(path)):
                orientation_chain[j] = orientation_chain[j - 1] * rotation_chain[j]
                position_chain[j] = np.dot(orientation_chain[j - 1].as_matrix(), offset_chain[j]) + position_chain[j - 1]
            distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))


    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = position_chain[i]
        if index in path2:
            joint_rotations[index] = rotation_chain[i].inv().as_quat()
        else:
            joint_rotations[index] = rotation_chain[i].as_quat()

    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * orientation_chain[0]).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = orientation_chain[root_index].as_quat()
            joint_positions[0] = position_chain[root_index]


    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])


    return joint_positions, joint_orientations

def part1_inverse_kinematics_torch(meta_data, joint_positions, joint_orientations, target_pose):
    """
    Inputs: 
        meta_data: MetaData class
        joint_positions: (M, 3)
        joint_orientations: (M, 4)
        target_pose: (3, )
    Outputs:
        After IK
        joint_positions: (M, 3)
        joint_orientations: (M, 4)
    """

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #
    if len(path2) == 1:
        path2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations(joint_name, joint_orientations, joint_parent)
    joint_offsets = get_joint_offsets(joint_name, joint_positions, joint_parent, joint_initial_position)


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_chain = np.empty((len(path), 3), dtype=float)
    offset_chain = np.empty((len(path), 3), dtype=float)

    # 对chain进行初始化
    if len(path2) > 1:
        rotation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv().as_euler('XYZ')
    else:
        rotation_chain[0] = R.from_quat(joint_orientations[path[0]]).as_euler('XYZ')

    start_position = torch.tensor(joint_positions[path[0]], requires_grad=False)
    offset_chain[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        index = path[i]
        if index in path2:
            # essential
            rotation_chain[i] = R.from_quat(joint_rotations[path[i]]).inv().as_euler('XYZ')
            offset_chain[i] = -joint_offsets[path[i - 1]]
            # essential
        else:
            rotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
            offset_chain[i] = joint_offsets[index]

    # pytorch autograde
    rotation_chain_tensor = torch.tensor(rotation_chain, requires_grad=True, dtype=torch.float32)
    offset_chain_tensor = torch.tensor(offset_chain, requires_grad=False, dtype=torch.float32)
    target_position = torch.tensor(target_pose, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_path = path.index(0)
    max_times = 50
    lr = 0.1
    while max_times > 0:
        # 向前计算end position
        max_times -= 1
        cur_position = start_position
        cur_orientation = rotation_chain_tensor[0]
        for i in range(1, len(path)):
            cur_position = euler_angles_to_matrix(cur_orientation, 'XYZ') @ offset_chain_tensor[i] + cur_position
            orientation_matrix = euler_angles_to_matrix(cur_orientation, 'XYZ') @ euler_angles_to_matrix(rotation_chain_tensor[i], 'XYZ')
            cur_orientation = matrix_to_euler_angles(orientation_matrix, 'XYZ')
            # joint_positions[path[i]] = cur_position.detach().numpy()
            # joint_orientations[path[i]] = R.from_euler('XYZ', cur_orientation.detach().numpy()).as_quat()
        dist = torch.norm(cur_position - target_position)
        if dist < 0.01 or max_times == 0:
            break

        # 反向传播
        dist.backward()
        rotation_chain_tensor.grad[rootjoint_index_in_path].zero_()
        rotation_chain_tensor.data.sub_(rotation_chain_tensor.grad * lr)
        rotation_chain_tensor.grad.zero_()

    # return joint_positions, joint_orientations

    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        if index in path2:
            joint_rotations[index] = R.from_euler('XYZ', rotation_chain_tensor[i].detach().numpy()).inv().as_quat()
        else:
            joint_rotations[index] = R.from_euler('XYZ', rotation_chain_tensor[i].detach().numpy()).as_quat()


    # 当IK链不过rootjoint时，IK起点的rotation需要特殊处理
    if path2 == [] and path[0] != 0:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() 
                                    * R.from_euler('XYZ', rotation_chain_tensor[0].detach().numpy())).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if 0 in path and rootjoint_index_in_path != 0:
        rootjoint_pos = start_position
        rootjoint_ori = rotation_chain_tensor[0]
        for i in range(1, rootjoint_index_in_path + 1):
            rootjoint_pos = euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ offset_chain_tensor[i] + rootjoint_pos
            rootjoint_ori = matrix_to_euler_angles(euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ euler_angles_to_matrix(rotation_chain_tensor[i], 'XYZ'), 'XYZ')
        joint_orientations[0] = R.from_euler('XYZ', rootjoint_ori.detach().numpy()).as_quat()
        joint_positions[0] = rootjoint_pos.detach().numpy()

    
    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = np.array([relative_x + joint_positions[0][0], target_height, relative_z + joint_positions[0][2]])
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    metadata_right = task2_inverse_kinematics.MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'rWrist_end')

    # 2 chains for 2 control points
    lpath, path_name, lpath1, lpath2 = meta_data.get_path_from_root_to_end()
    rpath, path_name, rpath1, rpath2 = metadata_right.get_path_from_root_to_end()
    """
    lpath: ltoe -> pelvis -> lhand
    rpath: ltoe -> pelvis -> rhand
    lpath1: lhand -> pelvis; lpath2: ltoe -> pelvis
    rpath1: rhand -> pelvis; rpath2: ltoe -> pelvis
    """

    common_ancestor = 2
    # rpath has been reversed here, so don't need to reverse later
    # pelvis -> rhand
    # common ancester to alleviate repeated joints.
    rpath1 = list(reversed(rpath1))[common_ancestor:]

    if len(lpath2) == 1:
        lpath2 = []
    if len(rpath2) == 1:
        rpath2 = []

    joint_rotations = get_joint_rotations(joint_name, joint_orientations, joint_parent)
    joint_offsets = get_joint_offsets(joint_name, joint_positions, joint_parent, joint_initial_position)

    lrotation_chain = np.empty((len(lpath), 3), dtype=float)
    loffset_chain = np.empty((len(lpath), 3), dtype=float)

    # lchain initialization, need to reverse
    if len(lpath2) > 1:
        lrotation_chain[0] = R.from_quat(joint_orientations[lpath2[1]]).inv().as_euler('XYZ')
    else:
        lrotation_chain[0] = R.from_quat(joint_orientations[lpath[0]]).as_euler('XYZ')

    loffset_chain[0] = np.array([0.,0.,0.])
    start_position = torch.tensor(joint_positions[lpath[0]], requires_grad=False)
    
    for i in range(1, len(lpath)):
        index = lpath[i]
        if index in lpath2:
            # essential
            lrotation_chain[i] = R.from_quat(joint_rotations[lpath[i]]).inv().as_euler('XYZ')
            loffset_chain[i] = -joint_offsets[lpath[i - 1]]
            # essential
        else:
            lrotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
            loffset_chain[i] = joint_offsets[index]


    rrotation_chain = np.empty((len(rpath1), 3), dtype=float)
    roffset_chain = np.empty((len(rpath1), 3), dtype=float)

    for i in range(len(rpath1)):
        index = rpath1[i]
        rrotation_chain[i] = R.from_quat(joint_rotations[index]).as_euler('XYZ')
        roffset_chain[i] = joint_offsets[index]

    # pytorch autograde
    lrotation_chain_tensor = torch.tensor(lrotation_chain, requires_grad=True, dtype=torch.float32)
    loffset_chain_tensor = torch.tensor(loffset_chain, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_lpath = lpath.index(0)

    rrotation_chain_tensor = torch.tensor(rrotation_chain, requires_grad=True, dtype=torch.float32)
    roffset_chain_tensor = torch.tensor(roffset_chain, requires_grad=False, dtype=torch.float32)
    rootjoint_index_in_rpath = rpath.index(0)

    left_target_position = torch.tensor(left_target_pose, requires_grad=False)
    right_target_position = torch.tensor(right_target_pose, requires_grad=False)

    max_times = 1000
    lr = 0.01
    ancestor_root_idx = 2
    while max_times > 0:
        # compute left dist
        max_times -= 1
        cur_position = start_position
        cur_orientation = lrotation_chain_tensor[0]
        for i in range(1, len(lpath)):
            cur_position = euler_angles_to_matrix(cur_orientation, 'XYZ') @ loffset_chain_tensor[i] + cur_position
            orientation_matrix = euler_angles_to_matrix(cur_orientation, 'XYZ') @ euler_angles_to_matrix(lrotation_chain_tensor[i], 'XYZ')
            cur_orientation = matrix_to_euler_angles(orientation_matrix, 'XYZ')
            if lpath[i] == ancestor_root_idx:
                # copy ancestor_root_idx's rotation and orientation
                # for the right chain to use.
                ca_orientation = cur_orientation.clone()
                ca_position = cur_position.clone()
        ldist = torch.norm(cur_position - left_target_position)

        # start from the ancestor_root_idx
        rcur_orientation = ca_orientation
        rcur_position = ca_position
        for i in range(len(rpath1)):
            rcur_position = euler_angles_to_matrix(rcur_orientation, 'XYZ') @ roffset_chain_tensor[i] + rcur_position
            rorientation_matrix = euler_angles_to_matrix(rcur_orientation, 'XYZ') @ euler_angles_to_matrix(rrotation_chain_tensor[i], 'XYZ')
            rcur_orientation = matrix_to_euler_angles(rorientation_matrix, 'XYZ')
        rdist = torch.norm(rcur_position - right_target_position)

        dist = ldist + rdist
        if dist < 0.01 or max_times == 0:
            break

        dist.backward()
        lrotation_chain_tensor.grad[rootjoint_index_in_lpath].zero_()
        # gradient descent update
        lrotation_chain_tensor.data.sub_(lrotation_chain_tensor.grad * lr)
        # clear gradient
        lrotation_chain_tensor.grad.zero_()

        rrotation_chain_tensor.data.sub_(rrotation_chain_tensor.grad * lr)
        rrotation_chain_tensor.grad.zero_()

    # left chain rotations back to joint rotations
    for i in range(len(lpath)):
        index = lpath[i]
        if index in lpath2:
            joint_rotations[index] = R.from_euler('XYZ', lrotation_chain_tensor[i].detach().numpy()).inv().as_quat()
        else:
            joint_rotations[index] = R.from_euler('XYZ', lrotation_chain_tensor[i].detach().numpy()).as_quat()

    # right chain rotations back to joint rotations
    for i in range(len(rpath1)):
        joint_rotations[rpath1[i]] = R.from_euler('XYZ', rrotation_chain_tensor[i].detach().numpy()).as_quat()


    if lpath2 == [] and lpath[0] != 0:
        joint_rotations[lpath[0]] = (R.from_quat(joint_orientations[joint_parent[lpath[0]]]).inv() 
                                    * R.from_euler('XYZ', lrotation_chain_tensor[0].detach().numpy())).as_quat()

    # what if 0 is not in lpath?
    if 0 in lpath and rootjoint_index_in_lpath != 0:
        # initialize root joint position and orientation
        rootjoint_pos = start_position
        rootjoint_ori = lrotation_chain_tensor[0]
        for i in range(1, rootjoint_index_in_lpath + 1):
            rootjoint_pos = euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ loffset_chain_tensor[i] + rootjoint_pos
            rootjoint_ori = matrix_to_euler_angles(euler_angles_to_matrix(rootjoint_ori, 'XYZ') @ euler_angles_to_matrix(lrotation_chain_tensor[i], 'XYZ'), 'XYZ')
        joint_orientations[0] = R.from_euler('XYZ', rootjoint_ori.detach().numpy()).as_quat()
        joint_positions[0] = rootjoint_pos.detach().numpy()
    
    
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])

    return joint_positions, joint_orientations