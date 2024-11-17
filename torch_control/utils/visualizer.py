import pathlib

import numpy as np

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_FOUND = True
except ImportError:
    print("WARNING: meshcat not installed. no vis supported.")
    MESHCAT_FOUND = False

class Visualizer:
    def __init__(self, num_drones: int = 1, spacing: float = 5.0):
        self.goal_pos = np.empty((num_drones, 3))
        self.goal_ang = np.empty((num_drones, 3))
        
        if not MESHCAT_FOUND:
            return

        self.vis = meshcat.Visualizer()
        self.vis.open()
        self.vis["/Cameras/default"].set_transform(
            tf.translation_matrix([0, 0, 0]).dot(tf.euler_matrix(0, np.radians(30), 0))
        )

        self.vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([-1, 0, 0])
        )

        meshpath = pathlib.Path(__file__).parent / "mesh" / "quadrotor_2.stl"
        quadrotor_mesh = g.StlMeshGeometry.from_file(meshpath)
        quadrotor_material = g.MeshLambertMaterial(color=0xffffff, transparent=False, opacity=0.5)
        
        # shape drones in near a square matrix
        length = int(np.ceil(np.sqrt(num_drones)))
        self.offsets = []
        for i in range(length):
            for j in range(length):
                self.offsets.append(np.array([i * spacing, j * spacing, 0]))
                if len(self.offsets) == num_drones:
                    break
            if len(self.offsets) == num_drones:
                break

        # make the center of the drones the origin
        self.offsets = np.array(self.offsets) - np.mean(self.offsets, axis=0)
        
        for i in range(num_drones):
            self.vis[f"drone_{i}"].set_object(quadrotor_mesh, quadrotor_material)
            
    def set_state(self, pos, rot, mode='quat'):
        if not MESHCAT_FOUND:
            return
        for i, (offset_i, pos_i, rot_i) in enumerate(zip(self.offsets, pos, rot)):
            if mode == 'quat':
                self.vis[f"drone_{i}"].set_transform(
                    tf.translation_matrix(pos_i + offset_i).dot(
                        tf.quaternion_matrix(rot_i).dot(tf.euler_matrix(0, 0, np.pi))))
            else:
                self.vis[f"drone_{i}"].set_transform(
                    tf.translation_matrix(pos_i + offset_i).dot(
                        tf.euler_matrix(*rot_i).dot(tf.euler_matrix(0, 0, np.pi))))

    def set_goal_state(self, goal_pos, goal_ang):
        self.goal_pos = goal_pos
        self.goal_ang = goal_ang

        # Visualize the goal state
        self.visualize_goal_state()

    def visualize_goal_state(self):
        if not MESHCAT_FOUND or self.goal_pos is None or self.goal_ang is None:
            return
        
        # Represent the goal as a semi-transparent quadrotor
        meshpath = pathlib.Path(__file__).parent / "mesh" / "quadrotor_2.stl"
        goal_mesh = g.StlMeshGeometry.from_file(meshpath)
        goal_material = g.MeshLambertMaterial(color=0xff0000, transparent=True, opacity=0.5)
        
        for i, (offset_i, pos_i, ang_i) in enumerate(zip(self.offsets, self.goal_pos, self.goal_ang)):
            self.vis[f"goal_{i}"].set_object(goal_mesh, goal_material)
            self.vis[f"goal_{i}"].set_transform(
                tf.translation_matrix(pos_i + offset_i).dot(tf.euler_matrix(*ang_i).dot(tf.euler_matrix(0, 0, np.pi))))