# environment/world.py
import pybullet as p
import pybullet_data
import random
import os


class World:
    """Monde PyBullet (gravite, sol) et generation d'une ville procedurale en URDF."""

    def __init__(self, physics_client_id: int):
        self.p = p
        self.physics_client_id = physics_client_id
        self.obstacle_ids: list[int] = []
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        self.p.setRealTimeSimulation(0, physicsClientId=self.physics_client_id)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def generate_city_urdf(self, config):

        filename = config.get("filename", "city")
        n_blocks_x = config.get("n_blocks_x", 4)
        n_blocks_y = config.get("n_blocks_y", 4)
        block_size = config.get("block_size", 20.0)
        road_width = config.get("road_width", 4.0)

        buildings_per_side = config.get("buildings_per_side", 3)
        # Un fichier par processus, pour pouvoir lancer plusieurs simulations en parallele
        import tempfile

        city_file = os.path.join(tempfile.gettempdir(), f"{filename}_{os.getpid()}_{id(self)}.urdf")
        self.city_urdf_path = city_file

        buildings = []

        with open(city_file, "w") as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<robot name="city">\n\n')

            # Sol
            total_size_x = n_blocks_x * (block_size + road_width)
            total_size_y = n_blocks_y * (block_size + road_width)

            f.write('  <link name="world_link"/>\n')
            f.write('  <link name="ground_plane">\n')
            f.write('    <visual>\n')
            f.write(f'      <geometry><box size="{total_size_x} {total_size_y} 0.1"/></geometry>\n')
            f.write('      <material name="asphalt"><color rgba="0.1 0.1 0.1 1"/></material>\n')
            f.write('    </visual>\n')
            f.write('    <collision>\n')
            f.write(f'      <geometry><box size="{total_size_x} {total_size_y} 0.1"/></geometry>\n')
            f.write('    </collision>\n')
            f.write('  </link>\n\n')

            f.write('  <joint name="ground_joint" type="fixed">\n')
            # Face superieure du sol a z = 0
            f.write('    <origin xyz="0 0 -0.05"/>\n')
            f.write('    <parent link="world_link"/><child link="ground_plane"/>\n')
            f.write('  </joint>\n\n')

            building_id = 0
            # Espacement entre les centres des blocs
            stride = block_size + road_width
            # Espacement entre les immeubles à l'intérieur d'un bloc
            inner_spacing = block_size / buildings_per_side

            for bx in range(n_blocks_x):
                for by in range(n_blocks_y):
                    block_center_x = (bx - n_blocks_x / 2.0) * stride + stride / 2.0
                    block_center_y = (by - n_blocks_y / 2.0) * stride + stride / 2.0

                    for ix in range(buildings_per_side):
                        for iy in range(buildings_per_side):
                            h = round(random.uniform(30.0, 50.0), 1)
                            # petit espace entre immeubles
                            w = inner_spacing * 0.9
                            d = inner_spacing * 0.9

                            # Position relative au centre du bloc
                            rel_x = (ix - buildings_per_side / 2.0) * inner_spacing + inner_spacing / 2.0
                            rel_y = (iy - buildings_per_side / 2.0) * inner_spacing + inner_spacing / 2.0

                            abs_x = block_center_x + rel_x
                            abs_y = block_center_y + rel_y
                            abs_z = h / 2.0

                            name = f"bld_{building_id}"
                            color = f"{round(random.uniform(0.3, 0.6), 2)} {round(random.uniform(0.3, 0.6), 2)} {round(random.uniform(0.3, 0.6), 2)} 1"

                            buildings.append(
                                {
                                    "id": building_id,
                                    "center": [abs_x, abs_y, h],
                                    "height": h,
                                    "width": w,
                                    "length": d,
                                }
                            )

                            mass = 100000.0
                            ixx = (1 / 12.0) * mass * (d**2 + h**2)
                            iyy = (1 / 12.0) * mass * (w**2 + h**2)
                            izz = (1 / 12.0) * mass * (w**2 + d**2)

                            f.write(f'  <link name="{name}">\n')

                            f.write('    <inertial>\n')
                            f.write('      <origin xyz="0 0 0" rpy="0 0 0"/>\n')
                            f.write(f'      <mass value="{mass}"/>\n')
                            f.write(
                                f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>\n'
                            )
                            f.write('    </inertial>\n')

                            f.write('    <visual>\n')
                            f.write(f'      <geometry><box size="{w} {d} {h}"/></geometry>\n')
                            f.write(
                                f'      <material name="mat_{building_id}"><color rgba="{color}"/></material>\n'
                            )
                            f.write('    </visual>\n')

                            f.write('    <collision>\n')
                            f.write(f'      <geometry><box size="{w} {d} {h}"/></geometry>\n')
                            f.write('    </collision>\n')
                            f.write('  </link>\n')

                            f.write(f'  <joint name="j_{name}" type="fixed">\n')
                            f.write('    <parent link="ground_plane"/>\n')
                            f.write(f'    <child link="{name}"/>\n')
                            f.write(f'    <origin xyz="{abs_x} {abs_y} {abs_z}" rpy="0 0 0"/>\n')
                            f.write('  </joint>\n\n')

                            building_id += 1

            f.write('</robot>\n')
        return buildings
