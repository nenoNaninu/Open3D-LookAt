import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math


def create_translation_mat(x: float, y: float, z: float):
    trans_mat = np.eye(4)
    trans_mat[:, 3] = [x, y, z, 1]

    return trans_mat


def create_scale_mat(x: float, y: float, z: float):
    scale_mat = np.eye(4)

    scale_mat[0, 0] = x
    scale_mat[1, 1] = y
    scale_mat[2, 2] = z

    return scale_mat


def create_look_at_mat(eye, target, up=[0.0, 1.0, 0.0]):
    # Open3D world coordinate system: right-handed coordinate system (Y up, same as OpenGL)
    # Open3D camera coordinate system: right-handed coordinate system (Y down, Z forward, same as OpenCV) 
    # https://github.com/intel-isl/Open3D/issues/1347

    eye = np.array(eye, dtype=np.float32, copy=True)
    target = np.array(target, dtype=np.float32, copy=True)
    up = np.array(up, dtype=np.float32, copy=True)

    z = eye - target
    z = z / np.linalg.norm(z)

    x = np.cross(up, z)
    x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    rotate_mat = np.array([
        [x[0], x[1], x[2], 0.0],
        [y[0], y[1], y[2], 0.0],
        [z[0], z[1], z[2], 0.0],
        [0, 0, 0, 1]
    ])

    trans_mat = create_translation_mat(-eye[0], -eye[1], -eye[2])

    scale_mat = create_scale_mat(1, -1, -1)

    tmp = np.dot(rotate_mat, trans_mat)
    tmp = np.dot(scale_mat, tmp)

    return tmp


if __name__ == "__main__":
    coordinate_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=640, height=480, visible=False)

    visualizer.add_geometry(coordinate_mesh_frame)

    view_control = visualizer.get_view_control()

    camera_params = view_control.convert_to_pinhole_camera_parameters()

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    sample_num = 100
    for idx in range(0, sample_num):

        theta = 2 * math.pi * idx / (sample_num - 1)
        x = math.cos(theta)
        z = math.sin(theta)

        extrinsic_mat = create_look_at_mat([x, 2, z], [0, 1, 0], [0, 1, 0])

        camera_params.extrinsic = extrinsic_mat

        view_control.convert_from_pinhole_camera_parameters(camera_params)

        camera_params = view_control.convert_to_pinhole_camera_parameters()

        image = visualizer.capture_screen_float_buffer(True)

        plt.imsave("./images/{:03d}.png".format(idx), np.asarray(image), dpi=1)